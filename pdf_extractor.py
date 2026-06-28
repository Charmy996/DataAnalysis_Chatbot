"""
PDF text + table extraction.
- Year extracted automatically from PDF content (not filename)
- Proper if __name__ == "__main__" guard
- bare except replaced with except Exception
- table rows converted to sentence chunks for RAG
- globally unique chunk IDs
"""
import fitz
import json
import os
import re
import hashlib

try:
    import pdfplumber
    import pandas as pd
    TABLES_ENABLED = True
except ImportError:
    TABLES_ENABLED = False

from config import OUTPUT_FOLDER, CHUNKS_FOLDER

os.makedirs(OUTPUT_FOLDER,  exist_ok=True)
os.makedirs(CHUNKS_FOLDER,  exist_ok=True)


def extract_year_from_name(name):
    """Fallback: try to get year from filename."""
    years = re.findall(r'\b(20\d{2}|19\d{2})\b', name)
    return years[0] if years else None


def extract_year_from_pdf(pdf_path):
    """
    Extract the report year from PDF content automatically.
    Looks for patterns like:
      - "for the year ended 31st March, 2025"
      - "Annual Report 2024-25"
      - "31 March 2025"
      - "FY2025"
    Checks first 5 pages and last 2 pages.
    Falls back to filename if nothing found.
    """
    try:
        doc = fitz.open(pdf_path)
        total = len(doc)
        pages_to_check = list(range(min(5, total))) + list(range(max(0, total - 2), total))

        candidates = []
        for i in pages_to_check:
            text = doc[i].get_text()

            # "year ended 31st March, 2025"
            for m in re.finditer(
                r'year\s+ended\s+(?:\d{1,2}(?:st|nd|rd|th)?\s+)?(?:march|mar)[\s,]+(\d{4})',
                text, re.IGNORECASE
            ):
                candidates.append(m.group(1))

            # "Annual Report 2024-25" -> take later year
            for m in re.finditer(r'annual\s+report\s+(\d{4})[–\-](\d{2,4})', text, re.IGNORECASE):
                y1, y2 = m.group(1), m.group(2)
                full_y2 = y1[:2] + y2 if len(y2) == 2 else y2
                candidates.append(full_y2)

            # "31st March, 2025"
            for m in re.finditer(r'31(?:st)?\s+march[\s,]+(\d{4})', text, re.IGNORECASE):
                candidates.append(m.group(1))

            # "FY2025" or "FY 2025"
            for m in re.finditer(r'\bfy\s?(\d{4})\b', text, re.IGNORECASE):
                candidates.append(m.group(1))

        if candidates:
            return max(set(candidates), key=candidates.count)

    except Exception as e:
        print(f"  Year extraction from content failed: {e}")

    # Fallback to filename
    return extract_year_from_name(os.path.splitext(os.path.basename(pdf_path))[0])


def make_chunk_id(company, tag, idx):
    raw = f"{company}_{tag}_{idx}"
    return hashlib.md5(raw.encode()).hexdigest()[:16] + f"_{company[:8]}"


def get_table_title(page, table_bbox):
    try:
        words = page.extract_words()
        table_top = table_bbox[1]
        above = [w for w in words
                 if w["bottom"] <= table_top and w["bottom"] >= table_top - 60]
        if above:
            above.sort(key=lambda w: (w["top"], w["x0"]))
            title = " ".join(w["text"] for w in above).strip()
            return re.sub(r"\s+", " ", title)[:150]
    except Exception:
        pass
    return ""


def is_numeric(val):
    if val is None:
        return False
    try:
        float(str(val).replace(",", "").replace("(", "-").replace(")", "").strip())
        return True
    except ValueError:
        return False


def table_row_to_sentence(row_dict, table_title, company, page_num):
    skip_cols = {"page", "company", "table_title"}
    label, numeric_pairs = None, []
    for key, val in row_dict.items():
        if key in skip_cols or val is None or str(val).strip() in ("", "None"):
            continue
        if label is None and not is_numeric(val):
            label = str(val).strip().replace("\n", " ")
        elif is_numeric(val):
            col = re.sub(r'^col_\d+$', '', str(key).strip())
            numeric_pairs.append((col, str(val).strip()))
    if not label or not numeric_pairs:
        return None
    parts = [f"{c}: {v}" if c else v for c, v in numeric_pairs]
    ctx = f"Table: {table_title}. " if table_title else ""
    return f"{ctx}{label} — {', '.join(parts)} (in Rupees crores)"


def extract_text(pdf_path, company, output_folder=OUTPUT_FOLDER):
    year = extract_year_from_pdf(pdf_path)
    print(f"  Detected year: {year}")
    doc  = fitz.open(pdf_path)
    pages = []
    for page_num, page in enumerate(doc):
        pages.append({
            "company": company,
            "year":    year,
            "page":    page_num + 1,
            "text":    page.get_text()
        })
    json_path = os.path.join(output_folder, f"{company}_text.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(pages, f, indent=2, ensure_ascii=False)
    print(f"  Text pages extracted: {len(pages)} (year={year})")
    return len(pages), year


def extract_tables(pdf_path, company, year, output_folder=OUTPUT_FOLDER):
    """Year passed in so text and table chunks share the same detected year."""
    if not TABLES_ENABLED:
        print("  pdfplumber/pandas not installed — skipping table extraction.")
        return 0, 0

    all_tables, sentence_chunks = [], []
    idx = 0

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            tables      = page.extract_tables()
            tables_bbox = page.find_tables()

            for t_idx, table in enumerate(tables):
                if not table or len(table) < 2:
                    continue
                try:
                    table_title = ""
                    if t_idx < len(tables_bbox):
                        table_title = get_table_title(page, tables_bbox[t_idx].bbox)

                    num_cols  = max(len(row) for row in table)
                    first_row = [str(c).strip() if c else "" for c in table[0]]
                    has_text  = any(
                        c and not c.replace(",","").replace(".","")
                        .replace("-","").replace(" ","").replace("(","")
                        .replace(")","").strip().isdigit()
                        for c in first_row
                    )

                    if has_text:
                        headers   = [h.replace("\n"," ").strip() or f"col_{i}"
                                     for i, h in enumerate(first_row)]
                        data_rows = table[1:]
                    else:
                        headers   = [f"col_{i}" for i in range(num_cols)]
                        data_rows = table

                    while len(headers) < num_cols:
                        headers.append(f"col_{len(headers)}")

                    cleaned = []
                    for row in data_rows:
                        row = list(row) + [None] * (num_cols - len(row))
                        cleaned.append([str(c).replace("\n"," ").strip() if c else None
                                        for c in row])

                    df = pd.DataFrame(cleaned, columns=headers)
                    df["page"]        = page_num + 1
                    df["company"]     = company
                    df["table_title"] = table_title or f"Table {t_idx+1} p{page_num+1}"
                    all_tables.append(df)

                    for _, row in df.iterrows():
                        sentence = table_row_to_sentence(
                            row.to_dict(),
                            table_title or f"Table p{page_num+1}",
                            company, page_num + 1
                        )
                        if sentence:
                            sentence_chunks.append({
                                "chunk_id": make_chunk_id(company, f"tbl_{page_num}", idx),
                                "company":  company,
                                "year":     year,
                                "page":     page_num + 1,
                                "label":    sentence.split(" — ")[0].replace(f"Table: {table_title}. ", "").strip(),
                                "word_count": len(sentence.split()),
                                "chunk":    sentence,
                                "source":   "table"
                            })
                            idx += 1

                except Exception as e:
                    print(f"  Skipped table on page {page_num+1}: {e}")

    if all_tables:
        final_df = pd.concat(all_tables, ignore_index=True, sort=False)
        final_df.to_csv(os.path.join(output_folder, f"{company}_tables.csv"),
                        index=False, encoding="utf-8")

    if sentence_chunks:
        path = os.path.join(CHUNKS_FOLDER, f"{company}_table_chunks.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(sentence_chunks, f, indent=2, ensure_ascii=False)
        print(f"  Tables: {len(all_tables)}, sentence chunks: {len(sentence_chunks)}")

    return len(all_tables), len(sentence_chunks)


def process_pdf(pdf_path, company, output_folder=OUTPUT_FOLDER):
    """Full pipeline. Returns (pages, tables, table_chunks, year)."""
    pages, year = extract_text(pdf_path, company, output_folder)
    tables, table_chunks = extract_tables(pdf_path, company, year, output_folder)
    return pages, tables, table_chunks, year


if __name__ == "__main__":
    from config import REPORTS_FOLDER
    for pdf_file in os.listdir(REPORTS_FOLDER):
        if not pdf_file.lower().endswith(".pdf"):
            continue
        company  = os.path.splitext(pdf_file)[0]
        pdf_path = os.path.join(REPORTS_FOLDER, pdf_file)
        print(f"\nProcessing: {company}")
        process_pdf(pdf_path, company)
    print("\nAll PDFs processed!")
