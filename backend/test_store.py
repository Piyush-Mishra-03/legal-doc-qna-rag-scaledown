from pathlib import Path
from ingest_pipeline import ingest_pdf_for_chunks
from vector_store import store_chunks


pdf_dir = Path("storage/uploads")
files = list(pdf_dir.glob("*.pdf"))

if not files:
    print("No PDFs found.")
    exit()

pdf = files[0]

chunks = ingest_pdf_for_chunks(pdf)

store_chunks(chunks, pdf.name)

print("Stored chunks:", len(chunks))
