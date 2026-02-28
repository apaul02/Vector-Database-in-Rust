import numpy as np
import os

# Let's simulate a standard embedding size (like small BERT)
dimensions = 384
# 100,000 vectors * 384 dims * 4 bytes (f32) = ~153 Megabytes
num_vectors = 100_000 

print(f"Generating {num_vectors} vectors with {dimensions} dimensions...")

# Generate a massive matrix of random 32-bit floats
# We use standard normal distribution to simulate real embedding spreads
data = np.random.randn(num_vectors, dimensions).astype(np.float32)

file_path = "mock_data.bin"

# Write raw C-contiguous bytes directly to disk
data.tofile(file_path)

file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
print(f"Successfully wrote {file_path}")
print(f"Total File Size: {file_size_mb:.2f} MB")
