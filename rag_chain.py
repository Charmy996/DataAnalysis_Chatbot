"""
RAG chain — combines retrieval + generation.
Fixes:
- answer_question returns (answer, results) — variable named correctly
- async version for Chainlit
- company and year filters wired through
"""
import asyncio
from retrieve import retrieve
from llm import build_context, create_prompt
from llm_gemini import get_answer


def answer_question(question, company=None, year=None):
    """Synchronous. Use in run_chat.py."""
    results = retrieve(question, company=company, year=year)
    if not results:
        return "No relevant information found in reports.", []
    context = build_context(results)
    prompt  = create_prompt(question, context)
    answer  = get_answer(prompt)
    return answer, results


async def answer_question_async(question, company=None, year=None, history_context=""):
    """Async. Use in Chainlit app.py."""
    results = retrieve(question, company=company, year=year)
    if not results:
        return "No relevant information found in reports.", []
    context = build_context(results)
    prompt  = create_prompt(question, context, history_context=history_context)
    answer  = await asyncio.to_thread(get_answer, prompt)
    return answer, results
