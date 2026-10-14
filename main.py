import os
import sys
import argparse
from typing import Dict, Any, List, Optional

import open3d as o3d
import numpy as np
import torch
import torch.multiprocessing as mp
from tqdm import tqdm

from conquer3d.data_structure import TriangleMesh

# Suppress Open3D C++ logging noise to prevent stdout/stderr line displacement
o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)


def process_mesh_task(task_args: tuple) -> Dict[str, Any]:
    """Worker task to compute and save discrete differential curvature metrics (Gaussian, k1, k2)
    in separate .npy files for a single mesh.

    Args:
        task_args (tuple): (obj_file_path, save_path_gauss, save_path_k1, save_path_k2,
                            device_str, dtype_str, overwrite, strict, verbose)

    Returns:
        Dict[str, Any]: Status dictionary containing result status, file path, and optional message.
    """
    (
        obj_file_path,
        save_path_gauss,
        save_path_k1,
        save_path_k2,
        device_str,
        dtype_str,
        overwrite,
        strict,
        verbose,
    ) = task_args

    if (
        os.path.exists(save_path_gauss)
        and os.path.exists(save_path_k1)
        and os.path.exists(save_path_k2)
        and not overwrite
    ):
        return {
            "status": "skipped",
            "path": obj_file_path,
            "msg": f"Curvature files for '{obj_file_path}' already exist. Skipping." if verbose else None,
        }

    try:
        # Load mesh via Open3D
        o3d_mesh = o3d.io.read_triangle_mesh(obj_file_path)
        vs_np = np.asarray(o3d_mesh.vertices)
        fs_np = np.asarray(o3d_mesh.triangles)

        # Validate geometric integrity
        if len(vs_np) == 0 or len(fs_np) == 0:
            return {
                "status": "empty",
                "path": obj_file_path,
                "msg": f"Warning: Mesh '{obj_file_path}' contains 0 vertices or 0 faces. Skipping." if verbose else None,
            }

        device = torch.device(device_str)
        vs = torch.from_numpy(vs_np).to(device, dtype=torch.float32)
        fs = torch.from_numpy(fs_np).to(device, dtype=torch.int32)

        # Construct conquer3d TriangleMesh topology
        mesh = TriangleMesh(vs, fs)

        # Differential geometry metrics (Gaussian curvature and principal curvatures)
        gauss_curv = mesh.get_gaussian_curvature().cpu().numpy()
        k1_t, k2_t = mesh.get_principal_curvatures().unbind(1)
        k1 = k1_t.cpu().numpy()
        k2 = k2_t.cpu().numpy()

        if dtype_str == "float16":
            f16_min = np.finfo(np.float16).min
            f16_max = np.finfo(np.float16).max
            gauss_curv = np.clip(gauss_curv, f16_min, f16_max).astype(np.float16)
            k1 = np.clip(k1, f16_min, f16_max).astype(np.float16)
            k2 = np.clip(k2, f16_min, f16_max).astype(np.float16)
        else:
            gauss_curv = gauss_curv.astype(np.float32, copy=False)
            k1 = k1.astype(np.float32, copy=False)
            k2 = k2.astype(np.float32, copy=False)

        # Create output directory only when ready to save
        os.makedirs(os.path.dirname(save_path_gauss), exist_ok=True)

        msg = None
        if overwrite and os.path.exists(save_path_gauss) and verbose:
            msg = f"Overwriting existing curvature files for '{obj_file_path}'."

        # Save separate .npy files
        np.save(save_path_gauss, gauss_curv)
        np.save(save_path_k1, k1)
        np.save(save_path_k2, k2)

        # Explicit dereference for memory sanitation
        del vs, fs, mesh, gauss_curv, k1_t, k2_t, k1, k2

        return {"status": "success", "path": obj_file_path, "msg": msg}

    except Exception as e:
        if strict:
            raise RuntimeError(f"Error processing '{obj_file_path}': {e}")
        return {
            "status": "error",
            "path": obj_file_path,
            "error": str(e),
            "msg": f"Error processing '{obj_file_path}': {e}. Skipping." if verbose else None,
        }


def resolve_devices(user_devices: Optional[List[str]]) -> List[str]:
    """Resolves and validates compute devices for workers."""
    if user_devices is not None and len(user_devices) > 0:
        resolved = []
        for d in user_devices:
            d_clean = d.strip()
            if d_clean.isdigit():
                resolved.append(f"cuda:{d_clean}")
            else:
                resolved.append(d_clean)
        return resolved

    if torch.cuda.is_available():
        num_cuda = torch.cuda.device_count()
        return [f"cuda:{i}" for i in range(num_cuda)]

    return ["cpu"]


def main():
    parser = argparse.ArgumentParser(
        description="Compute discrete differential curvature metrics (Gaussian, k1, k2) for the ABC dataset."
    )
    parser.add_argument(
        "-d", "--data_dir", type=str, required=True, help="Path to the directory containing ABC dataset."
    )
    parser.add_argument(
        "-j", "--jobs", type=int, default=min(4, os.cpu_count() or 1),
        help="Number of parallel worker processes (default: min(4, CPU count))."
    )
    parser.add_argument(
        "--devices", type=str, nargs="+", default=None,
        help="CUDA device IDs (e.g. 'cuda:0' or '0 1'). Defaults to all available CUDA devices."
    )
    parser.add_argument(
        "--dtype", type=str, choices=["float16", "float32"], default="float16",
        help="Storage precision for curvature arrays in .npy files (default: float16)."
    )
    parser.add_argument(
        "--chunk", type=str, default=None,
        help="Optional filter: comma-separated list of chunk names to process (e.g. 'abc_0000_obj_v00')."
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose diagnostic logging."
    )
    parser.add_argument(
        "-s", "--strict", action="store_true", help="Enable strict mode. Stop on first error."
    )
    parser.add_argument(
        "-o", "--overwrite", action="store_true", help="Overwrite existing curvature files."
    )
    args = parser.parse_args()

    if not os.path.isdir(args.data_dir):
        print(f"Error: Directory '{args.data_dir}' does not exist.", file=sys.stderr)
        sys.exit(1)

    obj_dir_path = os.path.join(args.data_dir, "obj")
    if not os.path.isdir(obj_dir_path):
        print(f"Error: 'obj' directory '{obj_dir_path}' does not exist.", file=sys.stderr)
        sys.exit(1)

    save_dir = os.path.join(args.data_dir, "curvature")
    os.makedirs(save_dir, exist_ok=True)

    devices = resolve_devices(args.devices)
    if args.verbose:
        print(f"Worker jobs: {args.jobs} | Devices: {devices} | Precision: {args.dtype}")

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
        print(f"Found {len(obj_dirs)} object chunk directories to process in '{args.data_dir}'.")

    # Initialize multiprocessing context
    mp_context = mp.get_context("spawn")

    for obj_dir_name in obj_dirs:
        obj_dir = os.path.join(obj_dir_path, obj_dir_name)
        save_obj_dir = os.path.join(save_dir, obj_dir_name)

        # Collect tasks for this chunk
        tasks = []
        obj_subdirs = sorted([
            d for d in os.listdir(obj_dir) if os.path.isdir(os.path.join(obj_dir, d))
        ])

        for obj_subdir_name in obj_subdirs:
            obj_subdir = os.path.join(obj_dir, obj_subdir_name)
            save_obj_subdir = os.path.join(save_obj_dir, obj_subdir_name)

            obj_files = sorted([
                f for f in os.listdir(obj_subdir)
                if f.endswith(".obj") and os.path.isfile(os.path.join(obj_subdir, f))
            ])

            for obj_file in obj_files:
                obj_file_path = os.path.join(obj_subdir, obj_file)
                base_name = os.path.splitext(obj_file)[0]
                save_path_gauss = os.path.join(save_obj_subdir, f"{base_name}_gauss.npy")
                save_path_k1 = os.path.join(save_obj_subdir, f"{base_name}_k1.npy")
                save_path_k2 = os.path.join(save_obj_subdir, f"{base_name}_k2.npy")
                dev = devices[len(tasks) % len(devices)]
                task_args = (
                    obj_file_path,
                    save_path_gauss,
                    save_path_k1,
                    save_path_k2,
                    dev,
                    args.dtype,
                    args.overwrite,
                    args.strict,
                    args.verbose,
                )
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
                res = process_mesh_task(task)
                if args.verbose and res.get("msg"):
                    tqdm.write(res["msg"], file=sys.stdout)
                pbar.update(1)
        else:
            with mp_context.Pool(processes=args.jobs) as pool:
                chunksize = max(1, min(16, len(tasks) // (args.jobs * 4) or 1))
                for res in pool.imap_unordered(process_mesh_task, tasks, chunksize=chunksize):
                    if args.verbose and res.get("msg"):
                        tqdm.write(res["msg"], file=sys.stdout)
                    pbar.update(1)

        pbar.close()


if __name__ == "__main__":
    main()