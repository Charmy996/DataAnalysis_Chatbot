def build_context(results):
    context = ""
    for r in results:
        year_str = f" | Year: {r['year']}" if r.get("year") else ""
        context += f"""
Company: {r['company']}{year_str}
Page: {r['page']}
Source type: {r.get('source', 'text')}

{r['chunk']}

-------------------
"""
    return context


def create_prompt(question, context, history_context=""):
    return f"""You are a senior financial analyst assistant working with OCR-extracted text from scanned PDF reports.

- If the question contains a specific number, search the context for that EXACT number
  and explain what financial item it represents and in which year it appears.
- Numbers in brackets like (3,541.54) typically indicate negative values or outflows.

OCR ERROR HANDLING:
- Text may contain OCR errors where numbers are misread
- Common OCR errors: "AT 00" = "47.00", "AO" = "47", "5000" = "50.00"
- Use context clues to interpret corrupted values

YOUR CAPABILITIES:
- Answer direct factual questions from the context
- Perform calculations (revenue growth, profit margins, YoY changes, ratios)
- Make logical inferences (e.g. "Was it profitable?" → check if net profit/PAT is positive)
- Compare figures across periods or companies if data is available
- Identify trends from the numbers provided

ANSWERING RULES:
- Answer ONLY from the context provided
- Show calculations clearly when doing math
- Always cite: Company, Year, Page for each figure used
- For logical questions reason step by step
- If genuinely not found say: "Not available in reports."
- Keep answers concise but complete
- Do NOT say what a value is NOT — just state what it IS
- Do NOT repeat the question back or explain what you are about to do
- Lead with the direct answer, then cite source

{f"CONVERSATION HISTORY:{history_context}" if history_context else ""}

RETRIEVED CONTEXT:
{context}

QUESTION:
{question}

Think step by step before answering.
"""
