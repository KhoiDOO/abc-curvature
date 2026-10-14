import os
import sys
import json
import argparse
from typing import Dict, Any, List, Optional

import open3d as o3d
import numpy as np
import torch.multiprocessing as mp
from tqdm import tqdm

# Suppress Open3D C++ logging noise to prevent console line displacement
o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)


def analyze_single_mesh_task(task_args: tuple) -> Optional[Dict[str, Any]]:
    """Worker task to query vertex and face counts for a single 3D mesh.

    Args:
        task_args (tuple): (obj_file_path, data_dir, strict, verbose)

    Returns:
        Optional[Dict[str, Any]]: Dictionary containing file_path, num_vertices, num_faces,
                                  and optional diagnostic message.
    """
    obj_file_path, data_dir, strict, verbose = task_args
    rel_path = os.path.relpath(obj_file_path, data_dir)

    try:
        o3d_mesh = o3d.io.read_triangle_mesh(obj_file_path)
        vs_np = np.asarray(o3d_mesh.vertices)
        fs_np = np.asarray(o3d_mesh.triangles)

        num_vertices = int(len(vs_np))
        num_faces = int(len(fs_np))

        msg = None
        if (num_vertices == 0 or num_faces == 0) and verbose:
            msg = f"Warning: Mesh '{rel_path}' contains {num_vertices} vertices and {num_faces} faces."

        return {
            "file_path": rel_path,
            "num_vertices": num_vertices,
            "num_faces": num_faces,
            "msg": msg,
        }

    except Exception as e:
        if strict:
            raise RuntimeError(f"Error processing '{obj_file_path}': {e}")
        return {
            "file_path": rel_path,
            "num_vertices": 0,
            "num_faces": 0,
            "error": str(e),
            "msg": f"Error reading '{rel_path}': {e}. Setting counts to 0." if verbose else None,
        }


def main():
    parser = argparse.ArgumentParser(
        description="Query vertex and face counts for all meshes in the ABC dataset."
    )
    parser.add_argument(
        "-d", "--data_dir", type=str, required=True, help="Path to the directory containing ABC dataset."
    )
    parser.add_argument(
        "-j", "--jobs", type=int, default=min(16, os.cpu_count() or 1),
        help="Number of parallel worker processes (default: min(16, CPU count))."
    )
    parser.add_argument(
        "--chunk", type=str, default=None,
        help="Optional filter: comma-separated list of chunk names to process (e.g. 'abc_0000_obj_v00')."
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output path for the JSON file (default: mesh_analysis.json in data_dir)."
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose diagnostic logging."
    )
    parser.add_argument(
        "-s", "--strict", action="store_true", help="Enable strict mode. Stop on first error."
    )
    args = parser.parse_args()

    if not os.path.isdir(args.data_dir):
        print(f"Error: Directory '{args.data_dir}' does not exist.", file=sys.stderr)
        sys.exit(1)

    obj_dir_path = os.path.join(args.data_dir, "obj")
    if not os.path.isdir(obj_dir_path):
        print(f"Error: 'obj' directory '{obj_dir_path}' does not exist.", file=sys.stderr)
        sys.exit(1)

    output_file_path = args.output or os.path.join(args.data_dir, "mesh_analysis.json")

    obj_dirs = sorted([
        d for d in os.listdir(obj_dir_path) if os.path.isdir(os.path.join(obj_dir_path, d))
    ])

    if args.chunk:
        allowed_chunks = set([c.strip() for c in args.chunk.split(",")])
        obj_dirs = [d for d in obj_dirs if d in allowed_chunks]
        if len(obj_dirs) == 0:
            print(f"Error: No directories match chunk filter '{args.chunk}'.", file=sys.stderr)
            sys.exit(1)

    if args.verbose:
        print(f"Found {len(obj_dirs)} object chunk directories to analyze in '{args.data_dir}'.")
        print(f"Worker jobs: {args.jobs} | Target output: {output_file_path}")
 
    # Initialize multiprocessing context
    mp_context = mp.get_context("spawn")
    all_records: List[Dict[str, Any]] = []

    for obj_dir_name in obj_dirs:
        obj_dir = os.path.join(obj_dir_path, obj_dir_name)

        tasks = []
        obj_subdirs = sorted([
            d for d in os.listdir(obj_dir) if os.path.isdir(os.path.join(obj_dir, d))
        ])

        for obj_subdir_name in obj_subdirs:
            obj_subdir = os.path.join(obj_dir, obj_subdir_name)

            obj_files = sorted([
                f for f in os.listdir(obj_subdir)
                if f.endswith(".obj") and os.path.isfile(os.path.join(obj_subdir, f))
            ])

            for obj_file in obj_files:
                obj_file_path = os.path.join(obj_subdir, obj_file)
                task_args = (obj_file_path, args.data_dir, args.strict, args.verbose)
                tasks.append(task_args)

        if len(tasks) == 0:
            if args.verbose:
                print(f"No .obj files found in chunk '{obj_dir_name}'. Skipping.")
            continue

        chunk_desc = f"{obj_dir_name} ({len(tasks)} meshes)"

        pbar = tqdm(
            total=len(tasks),
            desc=chunk_desc,
            file=sys.stdout,
            dynamic_ncols=True,
            mininterval=0.1,
            leave=True,
        )

        if args.jobs <= 1:
            for task in tasks:
                res = analyze_single_mesh_task(task)
                if res is not None:
                    if args.verbose and res.get("msg"):
                        tqdm.write(res["msg"], file=sys.stdout)
                    all_records.append({
                        "file_path": res["file_path"],
                        "num_vertices": res["num_vertices"],
                        "num_faces": res["num_faces"],
                    })
                pbar.update(1)
        else:
            with mp_context.Pool(processes=args.jobs) as pool:
                chunksize = max(1, min(64, len(tasks) // (args.jobs * 4) or 1))
                for res in pool.imap_unordered(analyze_single_mesh_task, tasks, chunksize=chunksize):
                    if res is not None:
                        if args.verbose and res.get("msg"):
                            tqdm.write(res["msg"], file=sys.stdout)
                        all_records.append({
                            "file_path": res["file_path"],
                            "num_vertices": res["num_vertices"],
                            "num_faces": res["num_faces"],
                        })
                    pbar.update(1)

        pbar.close()

    # Deterministically sort results by file_path
    all_records.sort(key=lambda r: r["file_path"])

    # Ensure parent output directory exists and write JSON
    os.makedirs(os.path.dirname(os.path.abspath(output_file_path)), exist_ok=True)
    with open(output_file_path, "w", encoding="utf-8") as f:
        json.dump(all_records, f, indent=2)

    print(f"Successfully analyzed {len(all_records)} meshes. Saved to '{output_file_path}'.")


if __name__ == "__main__":
    main()
