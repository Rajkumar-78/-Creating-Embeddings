
import time
import numpy as np
from sentence_transformers import SentenceTransformer


sentences = [
    "I enjoy learning Python programming.",
    "The weather is sunny today.",
    "Machine learning helps computers find patterns.",
    "I need to reset my account password.",
    "The team meeting starts at nine.",
    "Fresh vegetables are good for health.",
    "Vector databases store numerical representations.",
    "I like listening to music in the evening."
]

print("Loading SentenceTransformer model...")

model = SentenceTransformer("all-MiniLM-L6-v2")


start = time.perf_counter()

embeddings = model.encode(
    sentences,
    convert_to_numpy=True
)

local_time = time.perf_counter() - start

print("\nTASK 1: CREATING EMBEDDINGS")

print("Local shape:", embeddings.shape)
print(f"Execution time: {local_time:.4f} seconds")


assert embeddings.shape == (8, 384)

print("\nFirst 5 numbers of first vector:")
print(embeddings[0][:5])

print("\nVector dimension:", len(embeddings[0]))


sentences_100 = (sentences * 13)[:100]

start = time.perf_counter()

embeddings_100 = model.encode(
    sentences_100,
    convert_to_numpy=True
)

time_100 = time.perf_counter() - start

print("\n100 SENTENCES COMPARISON")

print("Number of sentences:", len(sentences_100))
print("Embedding shape:", embeddings_100.shape)
print(f"Local execution time: {time_100:.4f} seconds")
print("API cost: $0.00")


print("\nEXPLANATION")

print("Each number represents a learned coordinate")
print("in a 384-dimensional semantic vector.")
print("Individual numbers do not have a fixed human meaning.")
print("Together, they represent the meaning of the sentence.")

print("\nLOCAL VS API - FOUR POINTS")

print("1. Local models are suitable for offline processing.")
print("2. Local models avoid per-request API charges.")
print("3. API models are convenient for managed scaling.")
print("4. Choose based on quality, privacy, speed and cost.")
