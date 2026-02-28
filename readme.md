# High-Throughput Out-of-Core Vector Database

A custom Approximate Nearest Neighbor (ANN) vector database built entirely from scratch. Designed to handle high-dimensional machine learning embeddings (like those from LLMs and Sentence Transformers), this engine is written in **Rust** for memory safety and bare-metal performance, and exposed seamlessly to **Python** via `PyO3`.

## 🚀 Overview

Modern AI applications rely heavily on Vector Databases for RAG (Retrieval-Augmented Generation). However, loading massive datasets (e.g., 50GB+ of float arrays) into standard RAM frequently causes Out-Of-Memory (OOM) crashes. 

This project solves that by bypassing standard heap allocation. It delegates memory management directly to the operating system using memory-mapped files (`mmap`), allowing the database to search through datasets vastly larger than the available system RAM, while a custom Hierarchical Navigable Small World (HNSW) graph ensures sub-millisecond search latencies.



## 🧠 Core Systems Architecture

This engine was built to demonstrate low-level systems engineering principles applied to modern AI workloads:

* **Out-of-Core Storage via OS Page Cache:** Instead of loading vectors into a Rust `Vec`, the engine memory-maps a binary file from the SSD. When the search algorithm requests a vector, the CPU triggers a Page Fault, and the OS dynamically loads that specific 4KB chunk into virtual memory.
* **Zero-Copy Deserialization:** Using `bytemuck`, raw bytes fetched from the disk are safely cast directly into `[f32]` slices. No heap allocations occur during vector retrieval.
* **HNSW Graph with Beam Search:** Implements a multi-layered graph index with `efConstruction` exploration and dynamic connection pruning, achieving $O(\log N)$ search complexity.
* **Bypassing the Python GIL:** The core logic is compiled into a native C-extension (`.so` / `.pyd`). Python merely passes the query across the Foreign Function Interface (FFI) boundary, allowing Rust to execute the heavy mathematical traversal without being blocked by Python's Global Interpreter Lock.



## 🛠️ Tech Stack

* **Core Engine:** Rust (`std`, `collections::BinaryHeap`)
* **Memory & I/O:** `memmap2` (Cross-platform mmap), `bytemuck` (Zero-copy casting)
* **Mathematical Operations:** `ordered-float` (NaN-safe float comparison), Squared L2 Distance
* **Python FFI:** `PyO3`, `maturin`

## ⚙️ Getting Started

### Prerequisites
You need **Rust**, **Python 3.8+**, and **Maturin** installed on your system.

### Installation & Build

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd vector_db_core
   ```
2. Set up an isolated Python virtual environment:
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3. Install the build tool and compile the Rust engine:
```bash
pip install maturin numpy
maturin develop --release
```

## Benchmarking
To test the raw speed of the `mmap` engine on your hardware, use the included Benchmarking scripts:
```bash
# 1. Generate a 150MB+ binary file of 100,000 random 384-dimensional vectors
python generate_mock_data.py

# 2. Run the out-of-core indexing and search speed test
python benchmark.py 
```

## Example

```python
import numpy as np
import my_vector_db

# 1. Initialize the engine mapped to a disk file
# Parameters: (file_path, dimensions, M, ef_construction)
db = my_vector_db.VectorDB("mock_data.bin", 384, 16, 32)

# 2. Build the HNSW graph index in memory (Indexing 20,000 vectors)
db.build_index(0, 20000)

# 3. Query the database
# Create a dummy query vector (in production, this comes from an ML model)
query_vector = np.random.randn(384).astype(np.float32).tolist()

# Search executes in compiled Rust (< 1ms)
results = db.search(query_vector, k=5)

print("Top 5 Nearest Neighbors:")
for rank, (vec_id, distance) in enumerate(results, 1):
    print(f"{rank}. Vector ID: {vec_id:<6} | L2 Distance: {distance:.4f}")
```
