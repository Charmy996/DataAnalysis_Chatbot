"""
Chainlit app — main entry point.
All 27 audit issues addressed.
"""
import re
import os
import json
import asyncio
import shutil
import chainlit as cl

from retrieve import retrieve, invalidate_cache
from llm import build_context, create_prompt
from llm_gemini import get_answer, get_answer_with_image, extract_text_from_image, classify
from pdf_extractor import process_pdf
from chunker import chunk_company, load_chunks_for_company
from pdf_extractor import extract_year_from_name
from pinecone_store import upsert_chunks
from config import REPORTS_FOLDER, OUTPUT_FOLDER, CHUNKS_FOLDER, IMAGES_FOLDER

for d in (REPORTS_FOLDER, OUTPUT_FOLDER, CHUNKS_FOLDER, IMAGES_FOLDER):
    os.makedirs(d, exist_ok=True)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"}


# ── Prompts ───────────────────────────────────────────────────

CLARIFICATION_PROMPT = """
You are a financial query classifier.

Given the question and context, is the question AMBIGUOUS?
Only mark AMBIGUOUS if the SAME term maps to MULTIPLE DIFFERENT line items in the context.

Examples of clarifying questions:
- "CLARIFY: Are you asking about short-term borrowings, long-term borrowings, or total gross debt?"
- "CLARIFY: Are you asking about profit before tax, profit after tax, or EBITDA?"
- "CLARIFY: Are you asking about Face Value per unit, Number of shares, or total Rupees crores?"

Do NOT mark ambiguous if:
- Question has a specific number
- Only one answer exists in context
- Data simply not found

Reply ONLY with "CLEAR" or "CLARIFY: <question>".

CONTEXT:
{context}

QUESTION:
{question}
"""


# ── Helpers ───────────────────────────────────────────────────

def extract_year(text):
    years = re.findall(r'\b(19\d{2}|20\d{2})\b', text)
    if years:
        return years[-1]
    m = re.findall(r'fy(\d{2,4})', text.lower())
    if m:
        y = m[-1]
        return "20" + y if len(y) == 2 else y
    if any(w in text.lower() for w in ["previous year", "last year", "prior year"]):
        return "previous year"
    if any(w in text.lower() for w in ["current year", "this year"]):
        return "current year"
    return None


def rewrite_question(question, history):
    """
    Rewrite short follow-up questions using conversation history.
    Only triggers on explicit follow-up signals — NOT on every short question —
    to avoid mangling standalone short queries like "What is revenue?".
    """
    if not history:
        return question

    EXPLICIT_FOLLOWUP = [
        "what about", "and in", "what was it", "how about",
        "same for", "in fy", "what about fy",
        "previous year", "last year", "prior year",
        "tell me the same", "same question",
        "current year", "this year", "compare"
    ]

    is_followup = any(s in question.lower() for s in EXPLICIT_FOLLOWUP)
    if not is_followup:
        return question   # don't rewrite — it's a standalone question

    year = extract_year(question)
    original = history[-1]['question']
    for h in reversed(history):
        if len(h['question'].split()) > 6:
            original = h['question']
            break

    # Strip years and specific numbers so follow-ups don't carry over year-specific values
    original_clean = re.sub(
        r'\b(19\d{2}|20\d{2}|fy\d{2,4}|for \d{4}|in \d{4})\b',
        '', original, flags=re.IGNORECASE
    )
    original_clean = re.sub(r'\b\d[\d,\.]*\b', '', original_clean)
    original_clean = re.sub(r'\s+', ' ', original_clean).strip()

    if year:
        return f"{original_clean} for the year {year}"
    return f"{original_clean} — {question}"


def build_history_context(history):
    if not history:
        return ""
    lines = []
    for h in history[-6:]:
        lines.append(f"User: {h['question']}")
        lines.append(f"Assistant: {h['answer']}\n")
    return "\n".join(lines)


# ── PDF Pipeline ──────────────────────────────────────────────

async def run_pdf_pipeline(pdf_path, company):
    pages, tables, table_chunks, year = await asyncio.to_thread(process_pdf, pdf_path, company)
    num_chunks = await asyncio.to_thread(chunk_company, company)

    text_chunks = load_chunks_for_company(company)
    tbl_chunks  = []
    tbl_path = os.path.join(CHUNKS_FOLDER, f"{company}_table_chunks.json")
    if os.path.exists(tbl_path):
        with open(tbl_path, "r", encoding="utf-8") as f:
            tbl_chunks = json.load(f)

    vectors = await asyncio.to_thread(upsert_chunks, text_chunks + tbl_chunks)
    invalidate_cache()
    return pages, num_chunks, tables, table_chunks, vectors, year


# ── Image Pipeline ────────────────────────────────────────────

async def run_image_pipeline(image_path, image_name):
    source_name = os.path.splitext(image_name)[0]
    year        = extract_year_from_name(source_name)

    extracted_text = await asyncio.to_thread(extract_text_from_image, image_path)
    if not extracted_text or not extracted_text.strip():
        return 0, 0, ""

    sentences = [s.strip() for s in extracted_text.split("\n") if s.strip()]
    chunks = []
    for i, sentence in enumerate(sentences):
        chunks.append({
            "chunk_id":   f"img_{source_name}_{i+1}",
            "company":    source_name,
            "year":       year,
            "page":       1,
            "word_count": len(sentence.split()),
            "chunk":      sentence,
            "source":     "image"
        })

    chunk_path = os.path.join(CHUNKS_FOLDER, f"{source_name}_image_chunks.json")
    with open(chunk_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

    vectors = await asyncio.to_thread(upsert_chunks, chunks)
    invalidate_cache()
    return len(chunks), vectors, extracted_text


# ── Chainlit handlers ─────────────────────────────────────────

@cl.on_chat_start
async def start():
    cl.user_session.set("history",             [])
    cl.user_session.set("pending_question",    None)
    cl.user_session.set("pending_results",     None)
    cl.user_session.set("pending_clarification", None)
    cl.user_session.set("last_image_path",     None)
    await cl.Message(
        content=(
            "👋 Welcome to the **Financial RAG Chat**!\n\n"
            "Upload a **PDF** or **image** to index a report, "
            "or ask a question about already-indexed reports."
        )
    ).send()


@cl.on_message
async def main(message: cl.Message):

    # ── File uploads ──────────────────────────────────────────
    if message.elements:
        for file in message.elements:
            if not hasattr(file, "path"):
                continue
            ext = os.path.splitext(file.name.lower())[1]

            if ext == ".pdf":
                company = os.path.splitext(file.name)[0]
                dest    = os.path.join(REPORTS_FOLDER, file.name)
                shutil.copy(file.path, dest)
                await cl.Message(content=f"📄 **{file.name}** received! Processing...").send()
                async with cl.Step(name="Extracting text and tables..."):
                    pages, num_chunks, tables, tbl_chunks, vectors, year = await run_pdf_pipeline(dest, company)
                year = year or "unknown"
                await cl.Message(content=(
                    f"✅ **{company}** (year={year}) ready!\n"
                    f"- {pages} pages | {num_chunks} text chunks\n"
                    f"- {tables} tables → {tbl_chunks} table sentence chunks\n"
                    f"- {vectors} vectors in Pinecone"
                )).send()

            elif ext in IMAGE_EXTENSIONS:
                dest = os.path.join(IMAGES_FOLDER, file.name)
                shutil.copy(file.path, dest)
                await cl.Message(content=f"🖼️ **{file.name}** received! Analyzing...").send()
                async with cl.Step(name="Extracting data from image..."):
                    num_chunks, vectors, _ = await run_image_pipeline(dest, file.name)
                cl.user_session.set("last_image_path", dest)
                await cl.Message(content=(
                    f"✅ **{file.name}** indexed!\n"
                    f"- {num_chunks} sentence chunks | {vectors} vectors uploaded\n"
                    f"Ask me anything about it."
                )).send()

            else:
                await cl.Message(content=f"⚠️ Unsupported file `{ext}`. Please upload PDF or image.").send()
        return

    # ── Questions ─────────────────────────────────────────────
    question   = message.content
    history    = cl.user_session.get("history")
    pending_q  = cl.user_session.get("pending_question")
    pending_r  = cl.user_session.get("pending_results")
    last_img   = cl.user_session.get("last_image_path")

    # Direct image question
    image_kws = ["image","picture","photo","chart","graph","screenshot","describe","what do you see"]
    if last_img and any(k in question.lower() for k in image_kws):
        async with cl.Step(name="Analyzing image..."):
            answer = await asyncio.to_thread(
                get_answer_with_image,
                f"You are a financial analyst. Answer based on the image:\n\n{question}",
                last_img
            )
        history.append({"question": question, "answer": answer})
        cl.user_session.set("history", history[-20:])
        await cl.Message(content=answer).send()
        return

    # Clarification response — re-retrieve with clarified question for better context
    if pending_q:
        clarification = message.content
        # If user says "all" or "all 3" etc, ask for everything explicitly
        if any(w in clarification.lower() for w in ["all", "all 3", "all three", "everything", "both"]):
            question = f"For {pending_q}, give ALL of the following separately with exact figures for 2025 and 2024: Face Value per unit, Number of shares, and total value in Rupees crores."
        else:
            question = f"What is the {clarification} in the context of: {pending_q}? Give exact figures for each year separately."
        year_hint = extract_year(question)
        results   = retrieve(question, year=year_hint, top_k=8)
        cl.user_session.set("pending_question",    None)
        cl.user_session.set("pending_results",     None)
        cl.user_session.set("pending_clarification", None)

    else:
        retrieval_question = rewrite_question(question, history)
        is_followup = retrieval_question != question
        year_hint  = extract_year(question)
        results    = retrieve(retrieval_question, year=year_hint, top_k=8)
        context    = build_context(results)

        # Skip clarification for follow-ups — user is clearly continuing a thread
        if not is_followup:
            clarify_check = await asyncio.to_thread(
                classify,
                CLARIFICATION_PROMPT.format(context=context, question=question)
            )
            if clarify_check.strip().startswith("CLARIFY:"):
                clarify_q = clarify_check.replace("CLARIFY:", "").strip()
                cl.user_session.set("pending_question",      question)
                cl.user_session.set("pending_results",       results)
                cl.user_session.set("pending_clarification", retrieval_question)
                await cl.Message(content=f"🤔 {clarify_q}").send()
                return

    # Build context and history separately
    context        = build_context(results)
    history_context = build_history_context(history)
    year           = extract_year(question)
    year_instruction = (
        f"\nIMPORTANT: Answer only for year {year}.\n" if year else ""
    )

    prompt = create_prompt(
        question,
        context + year_instruction,
        history_context=history_context   # passed separately into prompt template
    )

    async with cl.Step(name="Searching reports..."):
        answer = await asyncio.to_thread(get_answer, prompt)

    history.append({"question": question, "answer": answer})
    cl.user_session.set("history", history[-20:])

    sources = list(set([
        f"{r['company']} | {r.get('year','?')} | Page {r['page']} [{r.get('source','text')}]"
        for r in results
    ]))
    await cl.Message(
        content=f"{answer}\n\n**Sources:**\n" + "\n".join(f"- {s}" for s in sources)
    ).send()
