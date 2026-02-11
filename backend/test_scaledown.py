from backend.scaledown_client import compress_text_with_scaledown

text = """
This is a very long legal document test paragraph.
This text will be compressed using ScaleDown API.
"""

out = compress_text_with_scaledown(text)

print("Compressed text:")
print(out)
