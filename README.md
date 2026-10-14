---
license: mit
task_categories:
  - image-to-3d
  - other
tags:
  - 3d
  - geometry
  - cad
  - curvature
  - gaussian-curvature
  - principal-curvatures
  - discrete-differential-geometry
  - differential-geometry
  - mesh
  - triangle-mesh
  - point-cloud
  - surface-reconstruction
  - geometric-deep-learning
  - neural-rendering
  - normal-estimation
  - shape-analysis
  - cad-models
  - implicit-surfaces
  - signed-distance-fields
  - conquer3d
  - cuda
  - pytorch
  - abc-dataset
  - computer-vision
size_categories:
  - 100K<n<1M
pretty_name: "ABC Curvature: Large-Scale Discrete Differential Geometry Benchmark"
---

# ABC Curvature: Large-Scale Discrete Differential Geometry Benchmark

This dataset provides precomputed vertex-level discrete Gaussian ($K$), minimum ($k_1$), and maximum ($k_2$) principal curvature vectors across **751,407 production CAD models** (100 chunk archives) from the [NYU ABC Dataset](https://deep-geometry.github.io/abc-dataset/).

It is designed for geometric deep learning, 3D shape analysis, CAD feature extraction, normal estimation, neural implicit surface reconstruction (SDF/UDF), and differentiable geometry benchmarking.

---

## 1. Dataset Overview & Key Offerings

- **Scale**: 751,407 production-grade watertight CAD models.
- **Vertex-Level Continuous Signals**:
  - `*_gauss.npy`: Discrete Gaussian curvature ($K = k_1 \cdot k_2$) per mesh vertex.
  - `*_k1.npy`: Minimum principal curvature ($k_1$) per mesh vertex.
  - `*_k2.npy`: Maximum principal curvature ($k_2$) per mesh vertex.
- **Precision & Numerical Clamping**: Float16 arrays finite-range clamped to $[-65504.0, +65504.0]$ to eliminate floating-point infinities and NaNs from high-frequency CAD edge singularities.
- **Companion Indices**:
  - `curvature_investigation_report_mesh_index.json`: Fast $O(1)$ lookup index mapping each mesh to its curvature difficulty tier (Tiers 1–5), mean absolute curvature ($\overline{|K|}$), peak curvature ($K_{\max}$), and flat surface ratio ($f_{\text{flat}}$).
  - `mesh_analysis.json`: Topological census detailing vertex counts ($V$), face counts ($F$), and $F/V$ Euler characteristic ratios.

---

## 2. Repository Structure & File Organization

```
kohido/abc-curvature/
├── README.md                                          # This Dataset Card
├── curvature_investigation_report_mesh_index.json     # Per-mesh curvature lookup table (172 MB)
├── mesh_analysis.json                                 # Per-mesh topology census (114 MB)
└── curvature/                                         # Chunk archives directory (~55 GB)
    ├── abc_0000_obj_v00.7z
    ├── abc_0001_obj_v00.7z
    └── ...
    └── abc_0099_obj_v00.7z
```

Inside each `abc_XXXX_obj_v00.7z` archive, files follow the standard ABC subfolder structure:
```
abc_XXXX_obj_v00/
└── <model_id>/
    ├── <model_id>_<hash>_trimesh_000_gauss.npy
    ├── <model_id>_<hash>_trimesh_000_k1.npy
    └── <model_id>_<hash>_trimesh_000_k2.npy
```

---

## 3. Usage & Code Examples

### 1. Download Metadata & Specific Chunk Archives

```python
from huggingface_hub import hf_hub_download

# Download the curvature index (172 MB)
index_file = hf_hub_download(
    repo_id="kohido/abc-curvature",
    filename="curvature_investigation_report_mesh_index.json",
    repo_type="dataset"
)

# Download a specific chunk archive (~550 MB)
chunk_file = hf_hub_download(
    repo_id="kohido/abc-curvature",
    filename="curvature/abc_0000_obj_v00.7z",
    repo_type="dataset"
)
```

### 2. Inspect Model Curvature & Filter by Difficulty Tier

```python
import json

with open("curvature_investigation_report_mesh_index.json", "r") as f:
    curvature_index = json.load(f)

# Example entry lookup
sample_key = list(curvature_index.keys())[0]
print(sample_key, curvature_index[sample_key])
# Output:
# {
#   "tier": "Tier 3: Moderate Curved CAD",
#   "mean_abs_k": 0.9666,
#   "max_abs_k": 87.81,
#   "flat_ratio": 0.964,
#   "num_vertices": 1156522
# }

# Filter high-curvature evaluation set (Tier 4 & Tier 5 only)
complex_models = [
    rel_path for rel_path, meta in curvature_index.items()
    if meta["tier"] in ["Tier 4: High Curvature CAD", "Tier 5: Extreme / Freeform CAD"]
]
print(f"Total complex models: {len(complex_models):,}")
```

### 3. Load Curvature Vectors in NumPy / PyTorch

```python
import numpy as np
import torch

# Zero-copy memory mapped load
gauss_k = np.load("00000739_e1b0437a642942c8a4fd9f81_trimesh_000_gauss.npy", mmap_mode="r")

# Convert to PyTorch float32/float64 tensor for neural training
k_tensor = torch.from_numpy(gauss_k.astype(np.float32))
print("Curvature vector shape:", k_tensor.shape)  # (N_vertices,)
```

---

## 4. GPU Curvature Computation Engine: conquer3d

All discrete differential curvature vectors in this dataset were extracted at high throughput using **[conquer3d](https://github.com/KhoiDOO/conquer3d.git)**, a high-performance 3D computer vision and geometry processing library for PyTorch and CUDA.

### Key Highlights
- **GPU-Accelerated Differential Geometry**: Fast, vectorized computation of discrete Gaussian curvature and principal curvatures ($k_1, k_2$) directly on CUDA tensors.
- **Native PyTorch Integration**: Provides `TriangleMesh` with zero-copy memory transfers, eliminating CPU-GPU serialization bottlenecks.
- **High Throughput Scalability**: Enabled processing all 751,407 production CAD models with sub-second per-mesh execution on modern GPUs.

### Example: Computing Curvature with conquer3d

```python
import torch
from conquer3d.data_structure import TriangleMesh

# Load mesh vertices and triangle indices onto GPU
vs = torch.tensor(vertices, dtype=torch.float32, device="cuda")
fs = torch.tensor(faces, dtype=torch.int32, device="cuda")

# Construct conquer3d GPU TriangleMesh
mesh = TriangleMesh(vs, fs)

# Compute discrete Gaussian curvature and principal curvatures
gaussian_curv = mesh.get_gaussian_curvature()        # Shape: (N_vertices,)
k1, k2 = mesh.get_principal_curvatures().unbind(1)   # Shapes: (N_vertices,), (N_vertices,)
```

For more details and source code, visit:
👉 **[https://github.com/KhoiDOO/conquer3d.git](https://github.com/KhoiDOO/conquer3d.git)**
