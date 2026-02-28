import time
import numpy as np
import my_vector_db

print("--- ML Infrastructure Benchmark ---")
print("Initializing Vector Database mapping from SSD...")

# Initialize our Rust engine: (file, dimensions, M, ef_construction)
db = my_vector_db.VectorDB("mock_data.bin", 384, 16, 32)
print(f"Memory map successful. Discovered {db.len()} vectors on disk.")

# Build the index. 
# We'll index the first 20,000 vectors. The disk mmap means we don't 
# load them into RAM all at once; Rust pages them in exactly when needed.
num_to_index = 20_000
print(f"\nBuilding HNSW Index for {num_to_index} vectors...")
start_build = time.time()
db.build_index(0, num_to_index)
build_time = time.time() - start_build
print(f"Index built in {build_time:.3f} seconds.")

# Create a random mock embedding to act as our "query"
query = np.random.randn(384).astype(np.float32).tolist()

print("\nExecuting Vector Search...")
start_search = time.time()

# Search the Rust engine for the top 5 closest matches!
results = db.search(query, 5)

search_time = time.time() - start_search

print(f"Search completed in {search_time * 1000:.3f} milliseconds!")
print("Top 5 matches (ID, L2 Distance):")
for rank, (vec_id, distance) in enumerate(results, 1):
    print(f" {rank}. Vector ID: {vec_id:<6} | Distance: {distance:.4f}")
