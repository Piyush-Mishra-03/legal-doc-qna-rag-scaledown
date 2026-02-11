import fitz  # PyMuPDF


def extract_text_from_pdf(file_path: str) -> str:
    """
    Extract full text from a legal PDF document.
    """

    doc = fitz.open(file_path)

    all_text = []

    for page_index, page in enumerate(doc):
        text = page.get_text()
        if text:
            all_text.append(text)

    doc.close()

    return "\n".join(all_text)
