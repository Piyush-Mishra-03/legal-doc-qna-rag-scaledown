from typing import List


def chunk_text(
    text: str,
    chunk_size: int = 800,
    overlap: int = 100
) -> List[str]:

    if not text:
        return []

    chunks = []
    start = 0
    length = len(text)

    while start < length:
        end = start + chunk_size
        chunk = text[start:end]

        chunks.append(chunk)

        start = end - overlap

        if start < 0:
            start = 0

    return chunks
