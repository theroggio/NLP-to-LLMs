import os

pdf_path = "pdfs/"

import fitz
from tqdm.auto import tqdm

def formatter(text):
    clean = text.replace("\n"," ").strip()
    return clean

def read_pdf(path):
    doc = fitz.open(path)
    pages = []
    for page_number, page in tqdm(enumerate(doc)):
        text = page.get_text()
        text = formatter(text)
        pages.append({
                "page_number": page_number,
                "page_char_count": len(text),
                "page_word_count": len(text.split(" ")),
                "page_sentence_count_raw": len(text.split(". ")),
                "page_token_count": len(text)/4,
                "text": text
            })
    return pages

docs = read_pdf(pdf_path)

