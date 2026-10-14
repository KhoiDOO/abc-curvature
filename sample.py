#!/usr/bin/env python3
"""
sample.py

Samples high-curvature meshes from the ABC dataset based on precomputed
discrete Gaussian curvature statistics in curvature_investigation_report_mesh_index.json.

Partitions:
- q3: Top-K highest curvature meshes in Quintile 3 (P40 to P60)
- q4: Top-K highest curvature meshes in Quintile 4 (P60 to P80)
- q5_1: Top-K highest curvature meshes in Quintile 5 Tertile 1 (P80.00 to P86.67)
- q5_2: Top-K highest curvature meshes in Quintile 5 Tertile 2 (P86.67 to P93.33)
- q5_3: Top-K highest curvature meshes in Quintile 5 Tertile 3 (P93.33 to P100.00)

Exports sampled file paths to a JSON dictionary of lists and copies the physical
.obj mesh files to subdirectories in /mnt/data/kdata/abc/sample/<partition>/.
"""

import os
import sys
import json
import shutil
import argparse
from typing import Dict, List, Tuple, Any
import numpy as np
from tqdm import tqdm


def discover_default_index_path() -> str:
    """Finds the default path for curvature_investigation_report_mesh_index.json."""
    candidates = [
        "./curvature_investigation_report_mesh_index.json",
        "curvature_investigation_report_mesh_index.json",
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    return candidates[0]


def format_bytes(num_bytes: float) -> str:
    """Formats bytes into human-readable string."""
    for unit in ["B", "KB", "MB", "GB"]:
        if abs(num_bytes) < 1024.0:
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.2f} TB"


def main():
    parser = argparse.ArgumentParser(
        description="Sample high-curvature meshes from Quintiles 3, 4, and tertiles of Quintile 5."
    )
    parser.add_argument(
        "-i", "--index", type=str, default=None,
        help="Path to curvature_investigation_report_mesh_index.json"
    )
    parser.add_argument(
        "--obj_dir", type=str, default="/mnt/data/kdata/abc/obj",
        help="Path to the directory containing OBJ chunks (default: /mnt/data/kdata/abc/obj)."
    )
    parser.add_argument(
        "--out_dir", type=str, default="/mnt/data/kdata/abc/sample",
        help="Target folder to copy sampled meshes into (default: /mnt/data/kdata/abc/sample)."
    )
    parser.add_argument(
        "-o", "--output", type=str, default="sampled_meshes.json",
        help="Output JSON file path in current working directory (default: sampled_meshes.json)."
    )
    parser.add_argument(
        "-k", "--k_per_bin", type=int, default=20,
        help="Number of highest-curvature models to sample per partition (default: 20)."
    )
    parser.add_argument(
        "--no_copy", action="store_true",
        help="Skip copying mesh files to destination directories."
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable verbose output logging."
    )
    args = parser.parse_args()

    index_path = args.index or discover_default_index_path()
    if not os.path.isfile(index_path):
        print(f"Error: Curvature mesh index '{index_path}' not found.", file=sys.stderr)
        sys.exit(1)

    print("=" * 75)
    print("ABC Dataset High-Curvature Stratified Sampling Pipeline")
    print(f"Index File:        {index_path}")
    print(f"Mesh Source Dir:   {args.obj_dir}")
    print(f"Staging Dest Dir:  {args.out_dir}")
    print(f"Output JSON File:  {os.path.abspath(args.output)}")
    print(f"Top-K per Bin:     {args.k_per_bin}")
    print("=" * 75)

    print(f"Loading curvature index from '{index_path}'...")
    with open(index_path, "r", encoding="utf-8") as f:
        index_data: Dict[str, Dict[str, Any]] = json.load(f)

    total_meshes = len(index_data)
    print(f"Successfully loaded {total_meshes:,} mesh records.")

    # Extract all mean_abs_k values for quantile calculation
    items = list(index_data.items())
    k_vals = np.array([v["mean_abs_k"] for _, v in items], dtype=np.float64)

    # Compute exact percentiles:
    # Quintile 3: P40 to P60
    # Quintile 4: P60 to P80
    # Quintile 5 divided into 3 tertiles: P80 to P86.67, P86.67 to P93.33, P93.33 to P100
    p_keys = [40.0, 60.0, 80.0, 80.0 + (20.0 / 3.0), 80.0 + (40.0 / 3.0), 100.0]
    p_cuts = np.percentile(k_vals, p_keys)

    partitions: List[Dict[str, Any]] = [
        {
            "key": "q3",
            "name": "Quintile 3 (P40 — P60)",
            "low": float(p_cuts[0]),
            "high": float(p_cuts[1]),
        },
        {
            "key": "q4",
            "name": "Quintile 4 (P60 — P80)",
            "low": float(p_cuts[1]),
            "high": float(p_cuts[2]),
        },
        {
            "key": "q5_1",
            "name": "Quintile 5 Part 1 (P80.00 — P86.67)",
            "low": float(p_cuts[2]),
            "high": float(p_cuts[3]),
        },
        {
            "key": "q5_2",
            "name": "Quintile 5 Part 2 (P86.67 — P93.33)",
            "low": float(p_cuts[3]),
            "high": float(p_cuts[4]),
        },
        {
            "key": "q5_3",
            "name": "Quintile 5 Part 3 (P93.33 — P100.00)",
            "low": float(p_cuts[4]),
            "high": float(p_cuts[5]),
        },
    ]

    print("\nCalculated Partition Boundaries (Mean Absolute Curvature mean |K|):")
    for p in partitions:
        print(f"  [{p['key']:<5}] {p['name']:<35} : ({p['low']:.4f}, {p['high']:.4f}]")

    # Sample top-K meshes for each partition
    sampled_results: Dict[str, List[str]] = {}
    copy_tasks: List[Tuple[str, str, str]] = []  # (source_obj, dest_obj, partition_key)

    print("\nSelecting Top-K Highest Curvature Meshes per Partition...")
    for p in partitions:
        p_key = p["key"]
        p_low = p["low"]
        p_high = p["high"]

        # Filter candidates strictly within this quantile interval
        candidates = [
            (rel_k, info["mean_abs_k"], info.get("num_vertices", 0))
            for rel_k, info in index_data.items()
            if p_low < info["mean_abs_k"] <= p_high
        ]

        # Sort descending by curvature
        candidates.sort(key=lambda x: x[1], reverse=True)
        top_k = candidates[:args.k_per_bin]

        if len(top_k) < args.k_per_bin:
            print(f"Warning: Partition '{p_key}' only contains {len(top_k)} meshes (requested {args.k_per_bin}).")

        # Map to mesh files
        sampled_mesh_paths: List[str] = []
        for rel_k, curv, n_verts in top_k:
            # rel_k format: abc_0000_obj_v00/00000739/00000739_..._trimesh_000_gauss.npy
            if rel_k.endswith("_gauss.npy"):
                rel_obj = rel_k[:-10] + ".obj"
            else:
                base, _ = os.path.splitext(rel_k)
                rel_obj = base + ".obj"

            src_obj_path = os.path.join(args.obj_dir, rel_obj)
            mesh_filename = os.path.basename(rel_obj)
            dst_obj_path = os.path.join(args.out_dir, p_key, mesh_filename)

            sampled_mesh_paths.append(dst_obj_path)
            copy_tasks.append((src_obj_path, dst_obj_path, p_key))

        sampled_results[p_key] = sampled_mesh_paths

        curv_min = top_k[-1][1] if top_k else 0.0
        curv_max = top_k[0][1] if top_k else 0.0
        print(f"  [{p_key:<5}] Sampled {len(top_k):2d} meshes (curvature range in sample: [{curv_min:.4f}, {curv_max:.4f}])")

    # Save output JSON
    print(f"\nWriting sample index JSON to '{args.output}'...")
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(sampled_results, f, indent=2)
    print(f"Saved {sum(len(v) for v in sampled_results.values())} mesh paths into '{args.output}'.")

    # Copy mesh files if requested
    if not args.no_copy:
        print(f"\nCopying {len(copy_tasks)} sampled mesh files to '{args.out_dir}'...")
        for p in partitions:
            os.makedirs(os.path.join(args.out_dir, p["key"]), exist_ok=True)

        copied_count = 0
        missing_count = 0
        total_copied_bytes = 0

        pbar = tqdm(
            copy_tasks,
            desc="Staging meshes",
            file=sys.stdout,
            dynamic_ncols=True,
            leave=True,
        )

        for src, dst, p_key in pbar:
            if not os.path.isfile(src):
                missing_count += 1
                if args.verbose:
                    tqdm.write(f"Warning: Source mesh not found: {src}", file=sys.stdout)
                continue

            try:
                # Use copyfile to prevent exFAT POSIX chmod/copystat permission errors
                shutil.copyfile(src, dst)
                copied_count += 1
                total_copied_bytes += os.path.getsize(dst)
            except Exception as e:
                if args.verbose:
                    tqdm.write(f"Error copying {src} -> {dst}: {e}", file=sys.stdout)

        pbar.close()

        print("\n" + "=" * 75)
        print("Copy Operation Summary")
        print("=" * 75)
        print(f"Total Meshes Scheduled: {len(copy_tasks)}")
        print(f"Successfully Staged:    {copied_count}")
        print(f"Missing Source Files:   {missing_count}")
        print(f"Total Staged Size:      {format_bytes(total_copied_bytes)}")
        print("=" * 75)

    print("\nSample Pipeline Finished Successfully!")


if __name__ == "__main__":
    main()
