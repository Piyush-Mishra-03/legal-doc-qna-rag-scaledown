from pathlib import Path
from ingest_pipeline import ingest_pdf_for_chunks

pdf_path = Path("storage/uploads")

files = list(pdf_path.glob("*.pdf"))

if not files:
    print("No PDF files found in storage/uploads")
    exit()

chunks = ingest_pdf_for_chunks(files[0])

print("Total chunks:", len(chunks))
print("Sample chunk:")
print("Original tokens:", chunks[0]["original_tokens"])
print("Compressed tokens:", chunks[0]["compressed_tokens"])
