import os
import sys
import json
import argparse
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import torch.multiprocessing as mp
from tqdm import tqdm


def compute_array_stats(arr: np.ndarray) -> Dict[str, Any]:
    """Computes comprehensive statistical moments and percentiles for a 1D numeric array."""
    n = len(arr)
    if n == 0:
        return {}

    arr_sorted = np.sort(arr)
    mean_val = float(np.mean(arr))
    std_val = float(np.std(arr))
    var_val = float(np.var(arr))
    min_val = float(np.min(arr))
    max_val = float(np.max(arr))

    # Higher-order moments
    if std_val > 0:
        z = (arr - mean_val) / std_val
        skewness = float(np.mean(z ** 3))
        kurtosis = float(np.mean(z ** 4)) - 3.0  # Excess kurtosis
    else:
        skewness = 0.0
        kurtosis = 0.0

    percentile_keys = [1, 5, 10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90, 95, 99, 99.9]
    percentile_vals = np.percentile(arr_sorted, percentile_keys)
    percentiles_dict = {f"p_{k}": float(v) for k, v in zip(percentile_keys, percentile_vals)}

    q1 = percentiles_dict["p_25"]
    q3 = percentiles_dict["p_75"]

    return {
        "count": n,
        "mean": mean_val,
        "std": std_val,
        "variance": var_val,
        "median": percentiles_dict["p_50"],
        "min": min_val,
        "max": max_val,
        "range": max_val - min_val,
        "q1": q1,
        "q3": q3,
        "iqr": q3 - q1,
        "skewness": skewness,
        "excess_kurtosis": kurtosis,
        "percentiles": percentiles_dict,
    }


def make_ascii_bar(pct: float, max_bar_len: int = 20) -> str:
    """Generates an ASCII bar representing a percentage value."""
    filled = int(round((pct / 100.0) * max_bar_len))
    return "█" * filled + "░" * (max_bar_len - filled)


def classify_curvature_tier(mean_abs_k: float) -> str:
    """Classifies a mesh into an intuitive geometric curvature tier based on surface bending energy."""
    if mean_abs_k < 0.01:
        return "Tier 1: Prismatic / Planar"
    elif mean_abs_k < 0.1:
        return "Tier 2: Subtle / Cylindrical"
    elif mean_abs_k < 1.0:
        return "Tier 3: Moderate Curved CAD"
    elif mean_abs_k < 10.0:
        return "Tier 4: High Curvature CAD"
    else:
        return "Tier 5: Extreme / Freeform CAD"


def analyze_curvature_task(task_args: tuple) -> Optional[Tuple]:
    """Worker task to read a Gaussian curvature vector and compute per-mesh reduction metrics.

    Args:
        task_args (tuple): (gauss_file_path, rel_path)

    Returns:
        Optional[Tuple]: Tuple containing per-mesh curvature descriptors:
            (rel_path, num_vertices, mean_abs_k, mean_signed_k, std_k, max_abs_k,
             flat_ratio, elliptic_ratio, hyperbolic_ratio, tier)
    """
    gauss_file_path, rel_path = task_args

    try:
        # Zero-copy memory mapped load
        arr = np.load(gauss_file_path, mmap_mode="r")
        n = len(arr)
        if n == 0:
            return None

        arr_f64 = arr.astype(np.float64)
        abs_arr = np.abs(arr_f64)

        mean_abs_k = float(np.mean(abs_arr))
        mean_signed_k = float(np.mean(arr_f64))
        std_k = float(np.std(arr_f64))
        max_abs_k = float(np.max(abs_arr))

        # Surface type decomposition (eps = 1e-4)
        n_flat = int(np.sum(abs_arr <= 1e-4))
        n_elliptic = int(np.sum(arr_f64 > 1e-4))
        n_hyperbolic = int(np.sum(arr_f64 < -1e-4))

        return (
            rel_path,
            n,
            mean_abs_k,
            mean_signed_k,
            std_k,
            max_abs_k,
            float(n_flat / n),
            float(n_elliptic / n),
            float(n_hyperbolic / n),
            classify_curvature_tier(mean_abs_k),
        )

    except Exception:
        return None


def generate_markdown_report(data: Dict[str, Any]) -> str:
    """Formats the statistical investigation into an executive Markdown report."""
    md = []
    g_stats = data["global_mean_abs_k"]
    max_stats = data["global_max_abs_k"]
    std_stats = data["global_std_k"]
    flat_stats = data["global_flat_ratio"]
    tiers = data["semantic_tiers"]
    quintiles = data["evaluation_quintiles"]
    chunk_summary = data["chunks_summary"]

    md.append("# ABC Dataset Discrete Gaussian Curvature Investigation Report\n")
    md.append("Multi-level structural, per-part, and global statistical audit across vertex-level curvature vectors.\n")

    # Executive Summary Card
    md.append("## 1. Executive Curvature Summary\n")
    md.append("| Metric | Value | Metric | Value |")
    md.append("| :--- | :--- | :--- | :--- |")
    md.append(f"| **Total Meshes Analyzed** | `{data['total_meshes']:,}` | **Total Chunks** | `{chunk_summary['total_chunks']}` |")
    md.append(f"| **Mean Surface Bending Energy (mean |K|)** | `{g_stats['mean']:.4f}` | **Median Surface Bending Energy** | `{g_stats['median']:.4f}` |")
    md.append(f"| **Mean Peak Sharpness (max |K|)** | `{max_stats['mean']:,.1f}` | **Median Peak Sharpness** | `{max_stats['median']:,.1f}` |")
    md.append(f"| **Average Flat Surface Ratio** | `{flat_stats['mean'] * 100.0:.2f}%` | **Median Flat Surface Ratio** | `{flat_stats['median'] * 100.0:.2f}%` |")
    md.append("")

    # Global Curvature Dispersion Table
    md.append("## 2. Global Curvature Distribution Moments\n")
    md.append("| Metric | Mean Absolute Curvature ($\\overline{|K|}$) | Peak Curvature ($K_{\\max}$) | Curvature Heterogeneity ($\\sigma_K$) | Flat Vertex Ratio ($f_{\\text{flat}}$) |")
    md.append("| :--- | :--- | :--- | :--- | :--- |")
    md.append(f"| **Mean ($\\mu$)** | {g_stats['mean']:.4f} | {max_stats['mean']:,.1f} | {std_stats['mean']:,.2f} | {flat_stats['mean']:.4f} |")
    md.append(f"| **Standard Deviation ($\\sigma$)** | {g_stats['std']:.4f} | {max_stats['std']:,.1f} | {std_stats['std']:,.2f} | {flat_stats['std']:.4f} |")
    md.append(f"| **Median ($P_{{50}}$)** | {g_stats['median']:.4f} | {max_stats['median']:,.1f} | {std_stats['median']:,.2f} | {flat_stats['median']:.4f} |")
    md.append(f"| **Interquartile Range (IQR)** | {g_stats['iqr']:.4f} | {max_stats['iqr']:,.1f} | {std_stats['iqr']:,.2f} | {flat_stats['iqr']:.4f} |")
    md.append(f"| **Min — Max Range** | {g_stats['min']:.4f} — {g_stats['max']:.4f} | {max_stats['min']:.1f} — {max_stats['max']:,.1f} | {std_stats['min']:.2f} — {std_stats['max']:,.2f} | {flat_stats['min']:.2f} — {flat_stats['max']:.2f} |")
    md.append(f"| **Skewness ($\\gamma_1$)** | {g_stats['skewness']:.2f} | {max_stats['skewness']:.2f} | {std_stats['skewness']:.2f} | {flat_stats['skewness']:.2f} |")
    md.append(f"| **Excess Kurtosis ($\\beta_2$)** | {g_stats['excess_kurtosis']:.2f} | {max_stats['excess_kurtosis']:.2f} | {std_stats['excess_kurtosis']:.2f} | {flat_stats['excess_kurtosis']:.2f} |")
    md.append("")

    # Quantile Stratification Table
    md.append("## 3. Quantiles & Percentiles Breakdown (Surface Bending Energy $\\overline{|K|}$)\n")
    md.append("| Percentile | Mean Absolute Curvature ($\\overline{|K|}$) | Peak Curvature ($K_{\\max}$) | Curvature Complexity Characterization |")
    md.append("| :---: | :--- | :--- | :--- |")
    gp = g_stats["percentiles"]
    mp_ = max_stats["percentiles"]
    md.append(f"| **$P_{{1}}$** | {gp['p_1']:.5f} | {mp_['p_1']:,.1f} | Nearly Flat / Prismatic Blocks |")
    md.append(f"| **$P_{{5}}$** | {gp['p_5']:.5f} | {mp_['p_5']:,.1f} | Very Mild Curvature Parts |")
    md.append(f"| **$P_{{10}}$** | {gp['p_10']:.4f} | {mp_['p_10']:,.1f} | Decile 1 ($D_1$) |")
    md.append(f"| **$P_{{25}}$ (Q1)** | {gp['p_25']:.4f} | {mp_['p_25']:,.1f} | Lower Quartile boundary |")
    md.append(f"| **$P_{{50}}$ (Median)** | {gp['p_50']:.4f} | {mp_['p_50']:,.1f} | Central Dataset Median |")
    md.append(f"| **$P_{{75}}$ (Q3)** | {gp['p_75']:.4f} | {mp_['p_75']:,.1f} | Upper Quartile boundary |")
    md.append(f"| **$P_{{90}}$** | {gp['p_90']:.4f} | {mp_['p_90']:,.1f} | Decile 9 ($D_9$) |")
    md.append(f"| **$P_{{95}}$** | {gp['p_95']:.4f} | {mp_['p_95']:,.1f} | High Surface Curvature Threshold |")
    md.append(f"| **$P_{{99}}$** | {gp['p_99']:.4f} | {mp_['p_99']:,.1f} | Top 1% Freeform Surfaces |")
    md.append(f"| **$P_{{99.9}}$** | {gp['p_99.9']:.4f} | {mp_['p_99.9']:,.1f} | Extreme Sharp Feature Outliers |")
    md.append("")

    # Semantic Geometric Complexity Tiers Table
    md.append("## 4. Semantic Curvature Complexity Tiers\n")
    md.append("Stratification based on physical surface bending energy (mean $|K|$):\n")
    md.append("| Tier Name | Range (mean $|K|$) | Mesh Count | % of Dataset | Cumulative % | Distribution |")
    md.append("| :--- | :--- | :--- | :--- | :--- | :--- |")
    for t in tiers:
        bar = make_ascii_bar(t["percentage"], max_bar_len=20)
        md.append(f"| **{t['label']}** | `{t['range']}` | {t['count']:,} | {t['percentage']:.2f}% | {t['cumulative_percentage']:.2f}% | `{bar}` |")
    md.append("")

    # Recommended Evaluation Quintiles Table
    md.append("## 5. Recommended Evaluation Partitions (Balanced Quintiles)\n")
    md.append("For balanced model evaluation across curvature difficulty levels:\n")
    md.append("| Split Tier | Curvature Range (mean $|K|$) | Target Population | Benchmark Purpose |")
    md.append("| :--- | :--- | :--- | :--- |")
    for q in quintiles:
        md.append(f"| **{q['split']}** | `{q['range']}` | ~20% ({q['count']:,}) | {q['purpose']} |")
    md.append("")

    # Local / Per-Chunk Diagnostics Table
    md.append("## 6. Local (Per-Chunk) Curvature Diagnostics\n")
    md.append(f"* **Total Chunk Archives Analyzed**: `{chunk_summary['total_chunks']}`")
    md.append(f"* **Average Models per Chunk**: `{chunk_summary['mean_meshes_per_chunk']:.1f}`\n")

    md.append("### Top 5 Most Curved Chunks (Highest Mean Surface Energy)")
    md.append("| Chunk Archive | Models | Mean $\\overline{|K|}$ | Median $\\overline{|K|}$ | Avg Peak $K_{\\max}$ | Flat Ratio |")
    md.append("| :--- | :--- | :--- | :--- | :--- | :--- |")
    for c in chunk_summary["top_5_curved"]:
        md.append(f"| `{c['chunk_id']}` | {c['count']:,} | {c['mean_abs_k']:.4f} | {c['median_abs_k']:.4f} | {c['mean_max_abs_k']:,.1f} | {c['mean_flat_ratio']*100:.1f}% |")
    md.append("")

    md.append("### Top 5 Flattest / Most Prismatic Chunks (Lowest Mean Surface Energy)")
    md.append("| Chunk Archive | Models | Mean $\\overline{|K|}$ | Median $\\overline{|K|}$ | Avg Peak $K_{\\max}$ | Flat Ratio |")
    md.append("| :--- | :--- | :--- | :--- | :--- | :--- |")
    for c in chunk_summary["top_5_flattest"]:
        md.append(f"| `{c['chunk_id']}` | {c['count']:,} | {c['mean_abs_k']:.4f} | {c['median_abs_k']:.4f} | {c['mean_max_abs_k']:,.1f} | {c['mean_flat_ratio']*100:.1f}% |")
    md.append("")

    return "\n".join(md)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze discrete Gaussian curvature vectors across ABC dataset meshes."
    )
    parser.add_argument(
        "-d", "--data_dir", type=str, required=True,
        help="Path to curvature directory containing chunk folders (e.g. /mnt/data/kdata/abc/curvature)."
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
        "-m", "--mesh_json", type=str, default=None,
        help="Optional path to mesh_analysis.json to bypass filesystem directory scanning."
    )
    parser.add_argument(
        "-o", "--output", type=str, default="curvature_investigation_report",
        help="Base path/filename for report outputs (default: 'curvature_investigation_report')."
    )
    parser.add_argument(
        "--no_index", action="store_true",
        help="Skip writing the per-mesh lookup index JSON."
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Print executive report to stdout upon completion."
    )
    args = parser.parse_args()

    if not os.path.isdir(args.data_dir):
        print(f"Error: Directory '{args.data_dir}' does not exist.", file=sys.stderr)
        sys.exit(1)

    # Discover chunk directories
    all_entries = sorted(os.listdir(args.data_dir))
    chunk_dirs = [
        d for d in all_entries
        if os.path.isdir(os.path.join(args.data_dir, d)) and not d.startswith(".")
    ]

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
        print(f"Error: No chunk directories found in '{args.data_dir}'.", file=sys.stderr)
        sys.exit(1)

    print("=" * 70)
    print("Discrete Gaussian Curvature Statistical Investigation")
    print(f"Source Directory:  {args.data_dir}")
    print(f"Worker Jobs:       {args.jobs}")
    print(f"Chunks Found:      {len(chunk_dirs)}")
    print("=" * 70)

    tasks: List[Tuple[str, str]] = []
    if args.mesh_json and os.path.isfile(args.mesh_json):
        print(f"Loading mesh inventory from '{args.mesh_json}'...")
        with open(args.mesh_json, "r", encoding="utf-8") as f:
            mesh_list = json.load(f)
        allowed_chunks = set(chunk_dirs)
        for item in mesh_list:
            orig_rel = item.get("file_path", "")
            if orig_rel.startswith("obj/"):
                orig_rel = orig_rel[4:]
            parts = orig_rel.split(os.sep)
            if not parts:
                continue
            chunk_name = parts[0]
            if chunk_name not in allowed_chunks:
                continue
            base, _ = orig_rel.rsplit(".", 1)
            gauss_rel = f"{base}_gauss.npy"
            gauss_full = os.path.join(args.data_dir, gauss_rel)
            tasks.append((gauss_full, gauss_rel))
        print(f"Mapped {len(tasks):,} tasks from inventory.")
    else:
        print("Scanning filesystem for curvature vector files...")
        for c_name in chunk_dirs:
            c_path = os.path.join(args.data_dir, c_name)
            for root, _, files in os.walk(c_path):
                for f in files:
                    if f.endswith("_gauss.npy"):
                        full_p = os.path.join(root, f)
                        rel_p = os.path.relpath(full_p, args.data_dir)
                        tasks.append((full_p, rel_p))

    if len(tasks) == 0:
        print("No '_gauss.npy' files found. Exiting.", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(tasks):,} Gaussian curvature files. Processing...")

    pbar = tqdm(
        total=len(tasks),
        desc="Analyzing curvature vectors",
        file=sys.stdout,
        dynamic_ncols=True,
        mininterval=0.1,
        leave=True,
    )

    mp_context = mp.get_context("spawn")
    raw_results: List[Tuple] = []

    if args.jobs <= 1:
        for t in tasks:
            res = analyze_curvature_task(t)
            if res is not None:
                raw_results.append(res)
            pbar.update(1)
    else:
        with mp_context.Pool(processes=args.jobs) as pool:
            chunksize = max(1, min(256, len(tasks) // (args.jobs * 4) or 1))
            for res in pool.imap_unordered(analyze_curvature_task, tasks, chunksize=chunksize):
                if res is not None:
                    raw_results.append(res)
                pbar.update(1)

    pbar.close()

    total_meshes = len(raw_results)
    print(f"Successfully analyzed {total_meshes:,} models. Computing multi-level statistics...")

    # Unpack columns
    file_paths = [r[0] for r in raw_results]
    num_vertices = [r[1] for r in raw_results]
    mean_abs_k = np.array([r[2] for r in raw_results], dtype=np.float64)
    mean_signed_k = np.array([r[3] for r in raw_results], dtype=np.float64)
    std_k = np.array([r[4] for r in raw_results], dtype=np.float64)
    max_abs_k = np.array([r[5] for r in raw_results], dtype=np.float64)
    flat_ratio = np.array([r[6] for r in raw_results], dtype=np.float64)
    elliptic_ratio = np.array([r[7] for r in raw_results], dtype=np.float64)
    hyperbolic_ratio = np.array([r[8] for r in raw_results], dtype=np.float64)
    tiers = [r[9] for r in raw_results]

    # Global Statistics
    g_mean_abs_stats = compute_array_stats(mean_abs_k)
    g_mean_signed_stats = compute_array_stats(mean_signed_k)
    g_std_stats = compute_array_stats(std_k)
    g_max_abs_stats = compute_array_stats(max_abs_k)
    g_flat_stats = compute_array_stats(flat_ratio)
    g_elliptic_stats = compute_array_stats(elliptic_ratio)
    g_hyperbolic_stats = compute_array_stats(hyperbolic_ratio)

    # Semantic Complexity Tiers
    semantic_bins = [
        (0.0, 0.01, "Tier 1: Prismatic / Planar"),
        (0.01, 0.1, "Tier 2: Subtle / Cylindrical"),
        (0.1, 1.0, "Tier 3: Moderate Curved CAD"),
        (1.0, 10.0, "Tier 4: High Curvature CAD"),
        (10.0, float("inf"), "Tier 5: Extreme / Freeform CAD"),
    ]
    total = len(mean_abs_k)
    semantic_tiers = []
    cumulative = 0.0
    for low, high, label in semantic_bins:
        mask = (mean_abs_k >= low) if high == float("inf") else ((mean_abs_k >= low) & (mean_abs_k < high))
        c_count = int(np.sum(mask))
        c_pct = (c_count / total) * 100.0 if total > 0 else 0.0
        cumulative += c_pct
        semantic_tiers.append({
            "label": label,
            "range": f"[{low:.2f}, {high if high == float('inf') else f'{high:.2f}'})",
            "count": c_count,
            "percentage": c_pct,
            "cumulative_percentage": min(100.0, cumulative),
        })

    # Recommended Evaluation Quintiles (20% slices based on mean_abs_k)
    q_cuts = np.percentile(mean_abs_k, [0, 20, 40, 60, 80, 100])
    quintiles = [
        {
            "split": "Quintile 1 (0-20%)",
            "range": f"[{q_cuts[0]:.4f} — {q_cuts[1]:.4f}]",
            "count": int(np.sum((mean_abs_k >= q_cuts[0]) & (mean_abs_k <= q_cuts[1]))),
            "purpose": "Near-flat / prismatic CAD baseline evaluation",
        },
        {
            "split": "Quintile 2 (20-40%)",
            "range": f"({q_cuts[1]:.4f} — {q_cuts[2]:.4f}]",
            "count": int(np.sum((mean_abs_k > q_cuts[1]) & (mean_abs_k <= q_cuts[2]))),
            "purpose": "Mild curvature / standard cylindrical and prismatic parts",
        },
        {
            "split": "Quintile 3 (40-60%)",
            "range": f"({q_cuts[2]:.4f} — {q_cuts[3]:.4f}]",
            "count": int(np.sum((mean_abs_k > q_cuts[2]) & (mean_abs_k <= q_cuts[3]))),
            "purpose": "Standard mechanical parts with fillets and curves",
        },
        {
            "split": "Quintile 4 (60-80%)",
            "range": f"({q_cuts[3]:.4f} — {q_cuts[4]:.4f}]",
            "count": int(np.sum((mean_abs_k > q_cuts[3]) & (mean_abs_k <= q_cuts[4]))),
            "purpose": "High-curvature / complex molded CAD components",
        },
        {
            "split": "Quintile 5 (80-100%)",
            "range": f"({q_cuts[4]:.4f} — {q_cuts[5]:.4f}]",
            "count": int(np.sum((mean_abs_k > q_cuts[4]) & (mean_abs_k <= q_cuts[5]))),
            "purpose": "Extreme freeform / organic surface stress-test",
        },
    ]

    # Per-Chunk / Local Statistics
    chunk_map: Dict[str, Dict[str, List[float]]] = {}
    for fp, mak, mxk, fr in zip(file_paths, mean_abs_k, max_abs_k, flat_ratio):
        chunk_id = fp.split(os.sep)[0]
        if chunk_id not in chunk_map:
            chunk_map[chunk_id] = {"mean_abs": [], "max_abs": [], "flat": []}
        chunk_map[chunk_id]["mean_abs"].append(mak)
        chunk_map[chunk_id]["max_abs"].append(mxk)
        chunk_map[chunk_id]["flat"].append(fr)

    chunk_stats_list = []
    for c_id, c_data in chunk_map.items():
        c_mean_abs = np.array(c_data["mean_abs"], dtype=np.float64)
        c_max_abs = np.array(c_data["max_abs"], dtype=np.float64)
        c_flat = np.array(c_data["flat"], dtype=np.float64)

        chunk_stats_list.append({
            "chunk_id": c_id,
            "count": len(c_mean_abs),
            "mean_abs_k": float(np.mean(c_mean_abs)),
            "median_abs_k": float(np.median(c_mean_abs)),
            "mean_max_abs_k": float(np.mean(c_max_abs)),
            "mean_flat_ratio": float(np.mean(c_flat)),
        })

    chunk_stats_sorted = sorted(chunk_stats_list, key=lambda x: x["mean_abs_k"])
    chunks_summary = {
        "total_chunks": len(chunk_stats_list),
        "mean_meshes_per_chunk": float(np.mean([c["count"] for c in chunk_stats_list])),
        "top_5_flattest": chunk_stats_sorted[:5],
        "top_5_curved": chunk_stats_sorted[-5:][::-1],
    }

    # Assemble complete metadata structure
    output_metadata = {
        "dataset_name": "ABC Dataset Discrete Gaussian Curvature Investigation",
        "total_meshes": total_meshes,
        "global_mean_abs_k": g_mean_abs_stats,
        "global_mean_signed_k": g_mean_signed_stats,
        "global_std_k": g_std_stats,
        "global_max_abs_k": g_max_abs_stats,
        "global_flat_ratio": g_flat_stats,
        "global_elliptic_ratio": g_elliptic_stats,
        "global_hyperbolic_ratio": g_hyperbolic_stats,
        "semantic_tiers": semantic_tiers,
        "evaluation_quintiles": quintiles,
        "chunks_summary": chunks_summary,
    }

    # Output file paths
    base_out = args.output
    if base_out.endswith(".json") or base_out.endswith(".md"):
        base_out = os.path.splitext(base_out)[0]

    json_path = f"{base_out}.json"
    md_path = f"{base_out}.md"
    index_path = f"{base_out}_mesh_index.json"

    print(f"Writing metadata report to '{json_path}'...")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output_metadata, f, indent=2)

    print(f"Writing formatted Markdown report to '{md_path}'...")
    md_content = generate_markdown_report(output_metadata)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    if not args.no_index:
        print(f"Writing per-mesh splitting lookup table to '{index_path}'...")
        # Format compact dictionary: file_path -> {tier, mean_abs_k, max_abs_k, flat_ratio, num_vertices}
        index_data = {
            fp: {
                "tier": t,
                "mean_abs_k": round(float(mak), 4),
                "max_abs_k": round(float(mxk), 2),
                "flat_ratio": round(float(fr), 4),
                "num_vertices": int(nv),
            }
            for fp, nv, mak, mxk, fr, t in zip(
                file_paths, num_vertices, mean_abs_k, max_abs_k, flat_ratio, tiers
            )
        }
        with open(index_path, "w", encoding="utf-8") as f:
            json.dump(index_data, f, indent=2)

    print("\n" + "=" * 70)
    print("Curvature Investigation Complete!")
    print(f"  Executive Report:  {os.path.abspath(md_path)}")
    print(f"  Summary Metadata:  {os.path.abspath(json_path)}")
    if not args.no_index:
        print(f"  Mesh Split Index:  {os.path.abspath(index_path)}")
    print("=" * 70)

    if args.verbose:
        print("\n" + md_content)


if __name__ == "__main__":
    main()
