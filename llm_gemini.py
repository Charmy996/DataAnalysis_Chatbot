"""
LLM calls — text and vision.
All model/client config comes from config.py.
"""
import base64
from config import groq_client, TEXT_MODEL, VISION_MODEL


def get_answer(prompt, max_tokens=2000):
    """Text-only LLM call. Higher max_tokens to avoid mid-sentence cutoff."""
    response = groq_client.chat.completions.create(
        model=TEXT_MODEL,
        messages=[
            {"role": "system", "content": "You are a senior financial analyst assistant."},
            {"role": "user",   "content": prompt}
        ],
        max_tokens=max_tokens
    )
    return response.choices[0].message.content


def get_answer_with_image(prompt, image_path=None, image_base64=None, media_type="image/jpeg"):
    """Vision LLM call. Pass image_path or pre-encoded image_base64."""
    if image_path and not image_base64:
        ext = image_path.lower().rsplit(".", 1)[-1]
        media_type = {
            "jpg": "image/jpeg", "jpeg": "image/jpeg",
            "png": "image/png",  "webp": "image/webp",
            "gif": "image/gif"
        }.get(ext, "image/jpeg")
        with open(image_path, "rb") as f:
            image_base64 = base64.b64encode(f.read()).decode("utf-8")

    response = groq_client.chat.completions.create(
        model=VISION_MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image_url",
                     "image_url": {"url": f"data:{media_type};base64,{image_base64}"}},
                    {"type": "text", "text": prompt}
                ]
            }
        ],
        max_tokens=2000
    )
    return response.choices[0].message.content


def extract_text_from_image(image_path=None, image_base64=None, media_type="image/jpeg"):
    """
    Extract all financial data from an image as structured sentences for RAG indexing.
    """
    prompt = """You are a financial data extraction assistant.
Extract ALL financial data from this image into clear structured sentences.
For every table row write: "<Row label> — <col name>: <value>, <col name>: <value> (in Rupees crores)"
For text paragraphs summarize key financial facts as sentences.
Include ALL numbers, percentages, dates, and company names.
Do NOT skip any data. Do NOT add commentary. Output only extracted sentences, one per line."""
    return get_answer_with_image(prompt, image_path=image_path,
                                 image_base64=image_base64, media_type=media_type)


def classify(prompt):
    """
    Lightweight classification call (e.g. CLEAR / CLARIFY).
    Separate from get_answer so each can be tuned independently.
    """
    response = groq_client.chat.completions.create(
        model=TEXT_MODEL,
        messages=[
            {"role": "system", "content": "You are a financial query classifier. Follow instructions exactly."},
            {"role": "user",   "content": prompt}
        ],
        max_tokens=200
    )
    return response.choices[0].message.content
