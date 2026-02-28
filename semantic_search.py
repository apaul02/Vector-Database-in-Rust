import numpy as np
from sentence_transformers import SentenceTransformer
import my_vector_db

# 1. Define our text knowledge base
documents = [
    "Rust is a blazing fast systems programming language.",
    "Python is the standard for machine learning and AI.",
    "A good biryani requires perfectly cooked basmati rice and marinated meat.",
    "The US and China are locked in a fierce semiconductor rivalry.",
    "Data structures and algorithms are essential for competitive programming.",
    "Running daily improves cardiovascular health and stamina."
]

print("Downloading and loading HuggingFace model...")
# all-MiniLM-L6-v2 is extremely fast and outputs exactly 384-dimensional vectors
model = SentenceTransformer('all-MiniLM-L6-v2')

print("Encoding documents into dense vectors...")
# Convert our strings into a NumPy matrix of floats
embeddings = model.encode(documents) 

# 2. Write the raw C-contiguous bytes to disk for our Rust mmap engine
db_file = "semantic_data.bin"
embeddings.astype(np.float32).tofile(db_file)

# 3. Initialize the CoreVec database
print("Initializing Rust HNSW Engine...")
db = my_vector_db.VectorDB(db_file, 384, 16, 32)
db.build_index(0, len(documents))

# 4. The Magic: Querying by Semantic Meaning
query_text = "Who is winning the microchip war?"
print(f"\nSearching for: '{query_text}'")

# Encode the human question into a vector
query_vector = model.encode(query_text).astype(np.float32).tolist()

# Send the vector through PyO3 into our Rust graph
results = db.search(query_vector, k=2)

print("\n--- Top Matches from CoreVec ---")
for rank, (vec_id, distance) in enumerate(results, 1):
    # We use the integer ID returned by Rust to look up the original text in Python
    matched_text = documents[vec_id]
    print(f"{rank}. [L2 Distance: {distance:.4f}] {matched_text}")
