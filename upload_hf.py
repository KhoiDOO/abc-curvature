#!/usr/bin/env python3
"""
upload_hf.py

Automated creation and upload pipeline for the ABC Curvature Dataset on Hugging Face.
Repository: kohido/abc-curvature (or user-specified repo_id)

Uploads:
1. README.md: Detailed Hugging Face Dataset Card with complete statistics, documentation, and usage examples.
2. curvature_investigation_report_mesh_index.json: Per-mesh curvature descriptors and semantic tier assignments.
3. mesh_analysis.json: Per-mesh topological census (vertices, faces, F/V ratios).
4. curvature/*.7z: 100 compressed 7z archives containing discrete Gaussian, min (k1), and max (k2) curvature vectors.

Ensures all file paths in JSON metadata files are strictly relative before uploading.
"""

import os
import sys
import json
import time
import argparse
from typing import Dict, List, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm
from huggingface_hub import HfApi, get_token, whoami


def build_dataset_readme(
    total_models: int = 751407,
    total_chunks: int = 100,
) -> str:
    """Constructs a comprehensive, publication-grade Hugging Face Dataset Card."""
    readme = f"""---
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

This dataset provides precomputed vertex-level discrete Gaussian ($K$), minimum ($k_1$), and maximum ($k_2$) principal curvature vectors across **{total_models:,} production CAD models** ({total_chunks} chunk archives) from the [NYU ABC Dataset](https://deep-geometry.github.io/abc-dataset/).

It is designed for geometric deep learning, 3D shape analysis, CAD feature extraction, normal estimation, neural implicit surface reconstruction (SDF/UDF), and differentiable geometry benchmarking.

---

## 1. Dataset Overview & Key Offerings

- **Scale**: {total_models:,} production-grade watertight CAD models.
- **Vertex-Level Continuous Signals**:
  - `*_gauss.npy`: Discrete Gaussian curvature ($K = k_1 \cdot k_2$) per mesh vertex.
  - `*_k1.npy`: Minimum principal curvature ($k_1$) per mesh vertex.
  - `*_k2.npy`: Maximum principal curvature ($k_2$) per mesh vertex.
- **Precision & Numerical Clamping**: Float16 arrays finite-range clamped to $[-65504.0, +65504.0]$ to eliminate floating-point infinities and NaNs from high-frequency CAD edge singularities.
- **Companion Indices**:
  - `curvature_investigation_report_mesh_index.json`: Fast $O(1)$ lookup index mapping each mesh to its curvature difficulty tier (Tiers 1–5), mean absolute curvature ($\overline{{|K|}}$), peak curvature ($K_{{\\max}}$), and flat surface ratio ($f_{{\\text{{flat}}}}$).
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
# {{
#   "tier": "Tier 3: Moderate Curved CAD",
#   "mean_abs_k": 0.9666,
#   "max_abs_k": 87.81,
#   "flat_ratio": 0.964,
#   "num_vertices": 1156522
# }}

# Filter high-curvature evaluation set (Tier 4 & Tier 5 only)
complex_models = [
    rel_path for rel_path, meta in curvature_index.items()
    if meta["tier"] in ["Tier 4: High Curvature CAD", "Tier 5: Extreme / Freeform CAD"]
]
print(f"Total complex models: {{len(complex_models):,}}")
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
"""
    return readme.strip() + "\n"


def verify_relative_paths(json_path: str, is_dict: bool = True) -> bool:
    """Verifies that all paths inside a JSON file are strictly relative.

    Args:
        json_path (str): Path to JSON file.
        is_dict (bool): True if JSON root is a dict (keys are paths), False if list of dicts.

    Returns:
        bool: True if all paths are relative, False otherwise.
    """
    print(f"Verifying relative path integrity in '{json_path}'...")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if is_dict:
        total = len(data)
        abs_count = sum(1 for k in data if k.startswith("/") or (len(k) > 1 and k[1] == ":"))
    else:
        total = len(data)
        abs_count = sum(
            1 for item in data
            if item.get("file_path", "").startswith("/") or (len(item.get("file_path", "")) > 1 and item.get("file_path", "")[1] == ":")
        )

    print(f"  Total records: {total:,} | Absolute paths found: {abs_count}")
    if abs_count > 0:
        print(f"Error: Found {abs_count} absolute paths in '{json_path}'!", file=sys.stderr)
        return False
    return True


def upload_single_file(
    api: HfApi,
    local_path: str,
    path_in_repo: str,
    repo_id: str,
    max_retries: int = 5,
) -> Tuple[str, bool, Optional[str]]:
    """Uploads a single file with exponential backoff retries.

    Returns:
        Tuple: (path_in_repo, success, error_message)
    """
    for attempt in range(1, max_retries + 1):
        try:
            api.upload_file(
                path_or_fileobj=local_path,
                path_in_repo=path_in_repo,
                repo_id=repo_id,
                repo_type="dataset",
            )
            return (path_in_repo, True, None)
        except Exception as e:
            if attempt == max_retries:
                return (path_in_repo, False, str(e))
            sleep_sec = min(60, 2 ** attempt)
            time.sleep(sleep_sec)
    return (path_in_repo, False, "Max retries exceeded")


def main():
    parser = argparse.ArgumentParser(
        description="Create and upload the ABC Curvature Dataset to Hugging Face."
    )
    parser.add_argument(
        "--repo_id", type=str, default="kohido/abc-curvature",
        help="Hugging Face dataset repository ID (default: 'kohido/abc-curvature')."
    )
    parser.add_argument(
        "--data_dir", type=str, required=True,
        help="Path to base data directory containing 'curvature_investigation_report_mesh_index.json' and 'mesh_analysis.json'."
    )
    parser.add_argument(
        "--curvature_dir", type=str, required=True,
        help="Path to directory containing the 100 .7z chunk archives."
    )
    parser.add_argument(
        "--index_json", type=str, default=None,
        help="Optional path override to curvature_investigation_report_mesh_index.json (defaults to <data_dir>/curvature_investigation_report_mesh_index.json)."
    )
    parser.add_argument(
        "--mesh_json", type=str, default=None,
        help="Optional path override to mesh_analysis.json (defaults to <data_dir>/mesh_analysis.json)."
    )
    parser.add_argument(
        "--private", action="store_true",
        help="Make repository private (default: False -> Public)."
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Simulate the upload without making any remote changes or data transfers."
    )
    parser.add_argument(
        "--num_workers", type=int, default=4,
        help="Number of concurrent upload worker threads (default: 4)."
    )
    parser.add_argument(
        "--chunk", type=str, default=None,
        help="Optional comma-separated list of chunk names to upload (e.g. 'abc_0000_obj_v00')."
    )
    parser.add_argument(
        "--skip_chunks", action="store_true",
        help="Upload metadata JSONs and README only, skipping the .7z chunk archives."
    )
    parser.add_argument(
        "--readme_out", type=str, default="README.md",
        help="Path to save generated Dataset Card locally (default: README.md)."
    )
    args = parser.parse_args()

    index_json_path = args.index_json or os.path.join(args.data_dir, "curvature_investigation_report_mesh_index.json")
    mesh_json_path = args.mesh_json or os.path.join(args.data_dir, "mesh_analysis.json")

    print("=" * 75)
    print("ABC Curvature Hugging Face Dataset Publishing Pipeline")
    print(f"Target Repo ID:    {args.repo_id}")
    print(f"Visibility:        {'Private' if args.private else 'Public'}")
    print(f"Data Dir:          {args.data_dir}")
    print(f"Curvature Dir:     {args.curvature_dir}")
    print(f"Mesh Index JSON:   {index_json_path}")
    print(f"Mesh Analysis JSON:{mesh_json_path}")
    print(f"Dry Run Mode:      {args.dry_run}")
    print(f"Parallel Workers:  {args.num_workers}")
    print("=" * 75)

    # 1. Authentication Check
    token = get_token()
    if not token:
        print("Error: No Hugging Face token found. Run 'huggingface-cli login' first.", file=sys.stderr)
        sys.exit(1)

    api = HfApi(token=token)
    user_info = whoami(token=token)
    user_name = user_info.get("name") or user_info.get("username", "Unknown")
    print(f"Authenticated as Hugging Face user: '{user_name}'")

    # 2. Pre-Flight Relative Path Audit
    if not os.path.isfile(index_json_path):
        print(f"Error: Index JSON '{index_json_path}' not found.", file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(mesh_json_path):
        print(f"Error: Analysis JSON '{mesh_json_path}' not found.", file=sys.stderr)
        sys.exit(1)

    valid_idx = verify_relative_paths(index_json_path, is_dict=True)
    valid_ma = verify_relative_paths(mesh_json_path, is_dict=False)
    if not (valid_idx and valid_ma):
        print("Error: Pre-flight path check failed. Refusing to upload non-relative paths.", file=sys.stderr)
        sys.exit(1)
    print("Pre-flight check passed: 100% of paths in JSON metadata files are strictly relative!\n")

    # 3. Generate Publication-Grade Dataset Card
    print(f"Generating publication-grade Dataset Card in '{args.readme_out}'...")
    readme_content = build_dataset_readme(total_models=751407, total_chunks=100)
    with open(args.readme_out, "w", encoding="utf-8") as f:
        f.write(readme_content)
    print("Dataset Card generated successfully.")

    # 4. Discover 7z Chunk Archives
    chunk_files: List[Tuple[str, str]] = []  # (local_path, remote_path)
    if not args.skip_chunks:
        if not os.path.isdir(args.curvature_dir):
            print(f"Error: Directory '{args.curvature_dir}' does not exist.", file=sys.stderr)
            sys.exit(1)

        all_7z = sorted([
            f for f in os.listdir(args.curvature_dir)
            if f.endswith(".7z") and f.startswith("abc_")
        ])

        if args.chunk:
            allowed = set(c.strip() if c.endswith(".7z") else f"{c.strip()}.7z" for c in args.chunk.split(","))
            all_7z = [f for f in all_7z if f in allowed]

        for f in all_7z:
            local_p = os.path.join(args.curvature_dir, f)
            remote_p = f"curvature/{f}"
            chunk_files.append((local_p, remote_p))

        print(f"Found {len(chunk_files)} .7z chunk archives in '{args.curvature_dir}'.")

    # Metadata files to upload to root
    metadata_uploads = [
        (args.readme_out, "README.md"),
        (index_json_path, "curvature_investigation_report_mesh_index.json"),
        (mesh_json_path, "mesh_analysis.json"),
    ]

    all_uploads = metadata_uploads + chunk_files
    total_upload_bytes = sum(os.path.getsize(lp) for lp, _ in all_uploads if os.path.isfile(lp))
    print(f"Total files scheduled for upload: {len(all_uploads)} ({total_upload_bytes / (1024**3):.2f} GB)")

    if args.dry_run:
        print("\n" + "=" * 75)
        print("[DRY RUN] Simulation complete. Files that would be uploaded:")
        for lp, rp in metadata_uploads:
            print(f"  [Metadata] {lp} -> repo:{rp} ({os.path.getsize(lp)/1024/1024:.1f} MB)")
        for lp, rp in chunk_files[:5]:
            print(f"  [Chunk]    {lp} -> repo:{rp} ({os.path.getsize(lp)/1024/1024:.1f} MB)")
        if len(chunk_files) > 5:
            print(f"  ... and {len(chunk_files) - 5} more .7z archives.")
        print(f"\nTarget Repository: https://huggingface.co/datasets/{args.repo_id}")
        print("=" * 75)
        return

    # 5. Create Dataset Repository on Hugging Face
    print(f"\nEnsuring dataset repository '{args.repo_id}' exists on Hugging Face...")
    repo_url = api.create_repo(
        repo_id=args.repo_id,
        repo_type="dataset",
        private=args.private,
        exist_ok=True,
    )
    print(f"Repository initialized: {repo_url}")

    # Query existing remote files to support resume
    print("Checking existing remote files in repository for resume...")
    try:
        remote_files = set(api.list_repo_files(repo_id=args.repo_id, repo_type="dataset"))
    except Exception:
        remote_files = set()
    print(f"Found {len(remote_files)} files already in remote repository.")

    # 6. Upload Metadata Files First
    print("\nUploading metadata files (README, indices)...")
    for lp, rp in metadata_uploads:
        print(f"  Uploading {rp} ({os.path.getsize(lp)/1024/1024:.1f} MB)...")
        _, success, err = upload_single_file(api, lp, rp, args.repo_id)
        if not success:
            print(f"Error uploading {rp}: {err}", file=sys.stderr)
            sys.exit(1)
        print(f"  Successfully uploaded {rp}.")

    # 7. Upload Chunk Archives
    if chunk_files:
        pending_chunks = [(lp, rp) for lp, rp in chunk_files if rp not in remote_files]
        skipped_count = len(chunk_files) - len(pending_chunks)
        if skipped_count > 0:
            print(f"Resuming: {skipped_count} chunks already exist remotely and will be skipped.")

        print(f"Uploading {len(pending_chunks)} remaining chunk archives (workers: {args.num_workers})...")
        pbar = tqdm(
            total=len(pending_chunks),
            desc="Uploading 7z chunks",
            file=sys.stdout,
            dynamic_ncols=True,
            leave=True,
        )

        errors: List[Tuple[str, str]] = []

        if args.num_workers <= 1:
            for lp, rp in pending_chunks:
                _, success, err = upload_single_file(api, lp, rp, args.repo_id)
                if not success:
                    errors.append((rp, err or "Unknown"))
                pbar.update(1)
        else:
            with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
                futures = {
                    executor.submit(upload_single_file, api, lp, rp, args.repo_id): rp
                    for lp, rp in pending_chunks
                }
                for fut in as_completed(futures):
                    rp, success, err = fut.result()
                    if not success:
                        errors.append((rp, err or "Unknown"))
                    pbar.update(1)

        pbar.close()

        if errors:
            print("\nWarning: Encounted errors on some chunk uploads:")
            for rp, err in errors:
                print(f"  - {rp}: {err}")
        else:
            print("\nAll chunk archives uploaded successfully!")

    print("\n" + "=" * 75)
    print("ABC Curvature Dataset Publishing Complete!")
    print(f"Dataset URL: https://huggingface.co/datasets/{args.repo_id}")
    print("=" * 75)


if __name__ == "__main__":
    main()
