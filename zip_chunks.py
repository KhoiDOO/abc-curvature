import os
import sys
import time
import shutil
import argparse
import subprocess
from typing import Dict, Any, List, Optional
import torch.multiprocessing as mp
from tqdm import tqdm




def format_bytes(num_bytes: float) -> str:
    """Format bytes into human-readable string (KB, MB, GB)."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(num_bytes) < 1024.0:
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.2f} PB"


def compress_chunk_task(task_args: tuple) -> Dict[str, Any]:
    """Worker task to compress a single chunk directory using the chosen format.

    Args:
        task_args (tuple): (chunk_name, chunk_path, out_archive_path, fmt, level, overwrite, verbose)

    Returns:
        Dict[str, Any]: Result dictionary containing status, paths, sizes, and elapsed time.
    """
    chunk_name, chunk_path, out_archive_path, fmt, level, overwrite, verbose = task_args

    if os.path.exists(out_archive_path) and not overwrite:
        return {
            "status": "skipped",
            "chunk_name": chunk_name,
            "archive_path": out_archive_path,
            "msg": f"Archive '{os.path.basename(out_archive_path)}' already exists. Skipping." if verbose else None,
        }

    # Stage temporary archive in fast RAM / tmpfs (/tmp) to avoid simultaneous read/write head thrashing on rotational HDD
    temp_archive = f"/tmp/{os.path.basename(out_archive_path)}.tmp_{os.getpid()}_{int(time.time())}"
    if os.path.exists(temp_archive):
        os.remove(temp_archive)

    os.makedirs(os.path.dirname(os.path.abspath(out_archive_path)), exist_ok=True)
    t0 = time.time()

    try:
        if fmt == "7z":
            has_7z = shutil.which("7z")
            if not has_7z:
                raise RuntimeError("Command '7z' not found. Please install p7zip/7z to use 7z format.")
            cmd = ["7z", "a", "-t7z", f"-mx={level}", "-mmt=on", temp_archive, "."]
            subprocess.run(cmd, cwd=chunk_path, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True)

        elif fmt == "zip":
            has_7z = shutil.which("7z")
            if has_7z:
                cmd = ["7z", "a", "-tzip", f"-mx={level}", "-mmt=on", temp_archive, "."]
                subprocess.run(cmd, cwd=chunk_path, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True)
            else:
                import zipfile
                with zipfile.ZipFile(temp_archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=level) as zf:
                    for root, _, files in os.walk(chunk_path):
                        for f in files:
                            full_p = os.path.join(root, f)
                            arc_p = os.path.relpath(full_p, chunk_path)
                            zf.write(full_p, arc_p)

        elif fmt == "tar.zst":
            cmd = ["tar", "--zstd", "-cf", temp_archive, "-C", chunk_path, "."]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True)

        elif fmt == "tar.gz":
            cmd = ["tar", "-czf", temp_archive, "-C", chunk_path, "."]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True)

        else:
            raise ValueError(f"Unsupported format: {fmt}")

        # Safely copy finished archive from /tmp to target location (safe on exFAT mounts)
        if os.path.exists(out_archive_path):
            os.remove(out_archive_path)
        shutil.copyfile(temp_archive, out_archive_path)
        try:
            os.remove(temp_archive)
        except OSError:
            pass

        elapsed = time.time() - t0
        compressed_size = os.path.getsize(out_archive_path)

        return {
            "status": "success",
            "chunk_name": chunk_name,
            "archive_path": out_archive_path,
            "compressed_size": compressed_size,
            "elapsed": elapsed,
            "msg": f"Compressed {chunk_name} ({format_bytes(compressed_size)}) in {elapsed:.1f}s" if verbose else None,
        }

    except Exception as e:
        if os.path.exists(temp_archive):
            try:
                os.remove(temp_archive)
            except OSError:
                pass
        return {
            "status": "error",
            "chunk_name": chunk_name,
            "error": str(e),
            "msg": f"Error compressing {chunk_name}: {e}",
        }


def main():
    parser = argparse.ArgumentParser(
        description="Compress ABC dataset chunk directories (e.g. abc_0000_obj_v00) into optimized archives."
    )
    parser.add_argument(
        "--expdir", type=str, required=True,
        help="Path to the directory containing chunk folders (e.g. /path/to/curvature)."
    )
    parser.add_argument(
        "--outdir", type=str, default=None,
        help="Destination directory for created archives (default: same as --expdir)."
    )
    parser.add_argument(
        "--format", type=str, choices=["7z", "zip", "tar.zst", "tar.gz"], default="7z",
        help="Compression format to use. '7z' achieves the highest compression ratio (~30%% raw size, matching ABC standard). Default: 7z."
    )
    parser.add_argument(
        "--level", type=int, default=None,
        help="Compression level (default: 3 for 7z [fast & high ratio], 6 for zip)."
    )
    parser.add_argument(
        "-j", "--jobs", type=int, default=1,
        help="Number of parallel chunk compression jobs (default: 1). On rotational spinning HDDs (e.g. exFAT /dev/sda), sequential 1 job prevents disk head thrashing."
    )
    parser.add_argument(
        "--chunk", type=str, default=None,
        help="Optional filter: comma-separated list of chunk names to process (e.g. 'abc_0000_obj_v00')."
    )
    parser.add_argument(
        "-o", "--overwrite", action="store_true",
        help="Overwrite existing archive files."
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable verbose diagnostic logging."
    )
    args = parser.parse_args()

    if not os.path.isdir(args.expdir):
        print(f"Error: Directory '{args.expdir}' does not exist.", file=sys.stderr)
        sys.exit(1)

    outdir = args.outdir or args.expdir
    os.makedirs(outdir, exist_ok=True)

    # Set default compression level per format
    if args.level is None:
        level = 3 if args.format == "7z" else 6
    else:
        level = args.level

    # Discover candidate chunk directories
    all_entries = sorted(os.listdir(args.expdir))
    chunk_dirs = [
        d for d in all_entries
        if os.path.isdir(os.path.join(args.expdir, d)) and not d.startswith(".") and d != os.path.basename(outdir)
    ]

    # Filter to valid chunk naming if applicable
    abc_chunks = [d for d in chunk_dirs if d.startswith("abc_")]
    if abc_chunks:
        chunk_dirs = abc_chunks

    if args.chunk:
        allowed = set(c.strip() for c in args.chunk.split(","))
        chunk_dirs = [d for d in chunk_dirs if d in allowed]
        if len(chunk_dirs) == 0:
            print(f"Error: No chunk directories match filter '{args.chunk}'.", file=sys.stderr)
            sys.exit(1)

    if len(chunk_dirs) == 0:
        print(f"Error: No chunk directories found in '{args.expdir}'.", file=sys.stderr)
        sys.exit(1)

    ext = f".{args.format}"
    print("=" * 70)
    print("ABC Dataset Chunk Compression Pipeline")
    print(f"Source Directory:  {args.expdir}")
    print(f"Output Directory:  {outdir}")
    print(f"Format:            {args.format.upper()} (extension: {ext})")
    print(f"Level:             {level}")
    print(f"Parallel Workers:  {args.jobs}")
    print(f"Chunks Found:      {len(chunk_dirs)}")
    print("=" * 70)

    # Prepare tasks
    tasks = []
    for c_name in chunk_dirs:
        c_path = os.path.join(args.expdir, c_name)
        archive_name = f"{c_name}{ext}"
        out_archive_path = os.path.join(outdir, archive_name)
        tasks.append((c_name, c_path, out_archive_path, args.format, level, args.overwrite, args.verbose))

    pbar = tqdm(
        total=len(tasks),
        desc=f"Compressing chunks ({args.format})",
        file=sys.stdout,
        dynamic_ncols=True,
        mininterval=0.1,
        leave=True,
    )

    mp_context = mp.get_context("spawn")
    results = []

    if args.jobs <= 1:
        for t in tasks:
            res = compress_chunk_task(t)
            if args.verbose and res.get("msg"):
                tqdm.write(res["msg"], file=sys.stdout)
            results.append(res)
            pbar.update(1)
    else:
        with mp_context.Pool(processes=args.jobs) as pool:
            for res in pool.imap_unordered(compress_chunk_task, tasks, chunksize=1):
                if args.verbose and res.get("msg"):
                    tqdm.write(res["msg"], file=sys.stdout)
                results.append(res)
                pbar.update(1)

    pbar.close()

    # Summary report
    successful = [r for r in results if r.get("status") == "success"]
    skipped = [r for r in results if r.get("status") == "skipped"]
    errors = [r for r in results if r.get("status") == "error"]

    total_compressed = sum(r.get("compressed_size", 0) for r in successful)
    total_time = sum(r.get("elapsed", 0) for r in successful)

    print("\n" + "=" * 70)
    print("Compression Summary Report")
    print("=" * 70)
    print(f"Total Chunks:           {len(tasks)}")
    print(f"Successfully Compressed:{len(successful)}")
    print(f"Skipped (Existing):     {len(skipped)}")
    print(f"Errors Encountered:     {len(errors)}")

    if successful:
        print(f"Total Compressed Size:  {format_bytes(total_compressed)}")
        print(f"Average Archive Size:   {format_bytes(total_compressed / len(successful))}")
        print(f"Total Processing Time:  {total_time:.1f}s (avg {total_time / len(successful):.1f}s/chunk)")

    if errors:
        print("\nErrors:")
        for e in errors:
            print(f"  - {e['chunk_name']}: {e.get('error')}")

    print("=" * 70)


if __name__ == "__main__":
    main()
