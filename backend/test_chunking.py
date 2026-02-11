from chunking import chunk_text

text = "This is a legal sentence. " * 200

chunks = chunk_text(text)

print("Total chunks:", len(chunks))
print(chunks[0][:200])
