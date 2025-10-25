# ML-Driven Loop Optimization Selector

A macOS-friendly prototype that **learns to pick loop optimizations** (vectorize, unroll, tile, tile+unroll) for classic kernels (matmul, 1D conv, 2D stencil).  
It compiles & benchmarks C kernels with `clang`, builds **static features** (sizes, loop depth, arithmetic intensity), and trains a small **ML model** to recommend the best optimization.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
