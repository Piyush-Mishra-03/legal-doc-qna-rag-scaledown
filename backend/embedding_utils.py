import os
import openai
import numpy as np

openai.api_key = os.getenv("UpYGg7q97W9CbWfctJ1XA3y6e06QwyZ18WhtjVLx")

def get_embedding(text: str, model="text-embedding-3-small"):
    resp = openai.Embedding.create(
        input=text,
        model=model
    )
    return np.array(resp["data"][0]["embedding"])

def cosine_similarity(a: np.ndarray, b: np.ndarray):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
