import os
import sys
import json
import argparse
from typing import Dict, Any, List, Tuple
import numpy as np


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
    sum_val = int(np.sum(arr))

    # Higher-order moments
    if std_val > 0:
        z = (arr - mean_val) / std_val
        skewness = float(np.mean(z ** 3))
        kurtosis = float(np.mean(z ** 4)) - 3.0  # Excess kurtosis
    else:
        skewness = 0.0
        kurtosis = 0.0

    # Percentiles
    percentile_keys = [1, 5, 10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90, 95, 99, 99.9]
    percentile_vals = np.percentile(arr_sorted, percentile_keys)
    percentiles_dict = {f"p_{k}": float(v) for k, v in zip(percentile_keys, percentile_vals)}

    q1 = percentiles_dict["p_25"]
    q3 = percentiles_dict["p_75"]
    iqr = q3 - q1

    return {
        "count": n,
        "sum": sum_val,
        "mean": mean_val,
        "std": std_val,
        "variance": var_val,
        "median": percentiles_dict["p_50"],
        "min": min_val,
        "max": max_val,
        "range": max_val - min_val,
        "q1": q1,
        "q3": q3,
        "iqr": iqr,
        "skewness": skewness,
        "excess_kurtosis": kurtosis,
        "zero_count": int(np.sum(arr == 0)),
        "percentiles": percentiles_dict,
    }


def compute_binned_distribution(arr: np.ndarray, bins: List[Tuple[float, float, str]]) -> List[Dict[str, Any]]:
    """Calculates counts and percentages for predefined interval bins."""
    total = len(arr)
    results = []
    cumulative = 0

    for low, high, label in bins:
        if high == float("inf"):
            mask = arr >= low
        else:
            mask = (arr >= low) & (arr < high)

        count = int(np.sum(mask))
        pct = (count / total) * 100.0 if total > 0 else 0.0
        cumulative += pct

        results.append({
            "label": label,
            "range": f"[{int(low):,}, {high if high == float('inf') else int(high):,})",
            "count": count,
            "percentage": pct,
            "cumulative_percentage": min(100.0, cumulative),
        })

    return results


def make_ascii_bar(pct: float, max_bar_len: int = 25) -> str:
    """Generates an ASCII bar representing a percentage value."""
    filled = int(round((pct / 100.0) * max_bar_len))
    return "█" * filled + "░" * (max_bar_len - filled)


def generate_markdown_report(data: Dict[str, Any]) -> str:
    """Formats the statistical investigation into an executive Markdown report."""
    md = []
    v_stats = data["vertices"]
    f_stats = data["faces"]
    r_stats = data["face_to_vertex_ratio"]
    tiers = data["semantic_tiers"]
    quintiles = data["evaluation_quintiles"]
    chunk_summary = data["chunks_summary"]

    md.append("# ABC Dataset 3D Mesh Topological Investigation Report\n")
    md.append("Comprehensive global and local structural audit across the entire CAD dataset.\n")

    # Executive Overview Cards
    md.append("## 1. Executive Dataset Summary\n")
    md.append("| Metric | Value | Metric | Value |")
    md.append("| :--- | :--- | :--- | :--- |")
    md.append(f"| **Total Meshes** | `{data['total_meshes']:,}` | **Total Chunks** | `{chunk_summary['total_chunks']}` |")
    md.append(f"| **Total Vertices** | `{v_stats['sum']:,}` | **Total Faces** | `{f_stats['sum']:,}` |")
    md.append(f"| **Mean Vertices / Mesh** | `{v_stats['mean']:,.1f}` | **Mean Faces / Mesh** | `{f_stats['mean']:,.1f}` |")
    md.append(f"| **Median Vertices** | `{v_stats['median']:,.1f}` | **Median Faces** | `{f_stats['median']:,.1f}` |")
    md.append(f"| **Degenerate (0 Verts)** | `{v_stats['zero_count']}` | **Degenerate (0 Faces)** | `{f_stats['zero_count']}` |")
    md.append("")

    # Statistical Moments Table
    md.append("## 2. Statistical Moments & Dispersion\n")
    md.append("| Property | Vertices ($V$) | Faces ($F$) | Ratio ($F/V$) |")
    md.append("| :--- | :--- | :--- | :--- |")
    md.append(f"| **Mean ($\mu$)** | {v_stats['mean']:,.2f} | {f_stats['mean']:,.2f} | {r_stats['mean']:.4f} |")
    md.append(f"| **Standard Deviation ($\sigma$)** | {v_stats['std']:,.2f} | {f_stats['std']:,.2f} | {r_stats['std']:.4f} |")
    md.append(f"| **Median ($P_{{50}}$)** | {v_stats['median']:,.1f} | {f_stats['median']:,.1f} | {r_stats['median']:.4f} |")
    md.append(f"| **Interquartile Range (IQR)** | {v_stats['iqr']:,.1f} | {f_stats['iqr']:,.1f} | {r_stats['iqr']:.4f} |")
    md.append(f"| **Min - Max Range** | {int(v_stats['min']):,} — {int(v_stats['max']):,} | {int(f_stats['min']):,} — {int(f_stats['max']):,} | {r_stats['min']:.2f} — {r_stats['max']:.2f} |")
    md.append(f"| **Skewness ($\gamma_1$)** | {v_stats['skewness']:.2f} | {f_stats['skewness']:.2f} | {r_stats['skewness']:.2f} |")
    md.append(f"| **Excess Kurtosis ($\\beta_2$)** | {v_stats['excess_kurtosis']:.2f} | {f_stats['excess_kurtosis']:.2f} | {r_stats['excess_kurtosis']:.2f} |")
    md.append("")
    md.append("> **Topology Insight**: A closed, watertight 2-manifold triangle mesh has $F \\approx 2V$ (Euler characteristic). The dataset median ratio is `" + f"{r_stats['median']:.4f}`" + ", demonstrating that the overwhelming majority of models exhibit valid 2-manifold triangle connectivity.\n")

    # Quantile Stratification Table
    md.append("## 3. Quantiles & Percentiles Breakdown\n")
    md.append("| Percentile | Vertices ($V$) | Faces ($F$) | Interpretation |")
    md.append("| :---: | :--- | :--- | :--- |")
    vp = v_stats["percentiles"]
    fp = f_stats["percentiles"]
    md.append(f"| **$P_{{1}}$** | {vp['p_1']:,.0f} | {fp['p_1']:,.0f} | Micro / Simplest Primitives |")
    md.append(f"| **$P_{{5}}$** | {vp['p_5']:,.0f} | {fp['p_5']:,.0f} | Tiny CAD Parts |")
    md.append(f"| **$P_{{10}}$** | {vp['p_10']:,.0f} | {fp['p_10']:,.0f} | Decile 1 ($D_1$) |")
    md.append(f"| **$P_{{25}}$ (Q1)** | {vp['p_25']:,.0f} | {fp['p_25']:,.0f} | Lower Quartile boundary |")
    md.append(f"| **$P_{{50}}$ (Median)** | {vp['p_50']:,.0f} | {fp['p_50']:,.0f} | Dataset Central Median |")
    md.append(f"| **$P_{{75}}$ (Q3)** | {vp['p_75']:,.0f} | {fp['p_75']:,.0f} | Upper Quartile boundary |")
    md.append(f"| **$P_{{90}}$** | {vp['p_90']:,.0f} | {fp['p_90']:,.0f} | Decile 9 ($D_9$) |")
    md.append(f"| **$P_{{95}}$** | {vp['p_95']:,.0f} | {fp['p_95']:,.0f} | High-Complexity Threshold |")
    md.append(f"| **$P_{{99}}$** | {vp['p_99']:,.0f} | {fp['p_99']:,.0f} | Ultra-Dense Top 1% Outliers |")
    md.append(f"| **$P_{{99.9}}$** | {vp['p_99.9']:,.0f} | {fp['p_99.9']:,.0f} | Extreme Top 0.1% Tail |")
    md.append("")

    # Semantic Geometric Tiers
    md.append("## 4. Semantic Geometric Complexity Tiers\n")
    md.append("| Tier Name | Vertex Range | Mesh Count | % of Dataset | Cumulative % | Distribution |")
    md.append("| :--- | :--- | :--- | :--- | :--- | :--- |")
    for t in tiers:
        bar = make_ascii_bar(t["percentage"], max_bar_len=20)
        md.append(f"| **{t['label']}** | `{t['range']}` | {t['count']:,} | {t['percentage']:.2f}% | {t['cumulative_percentage']:.2f}% | `{bar}` |")
    md.append("")

    # Recommended Evaluation Quintiles
    md.append("## 5. Recommended Evaluation Partitions (Balanced Quintiles)\n")
    md.append("For fair benchmarking, models should be partitioned into balanced slices so algorithms aren't evaluated solely on simple or massive models:\n")
    md.append("| Split Tier | Vertex Range ($V$) | Target Population | Evaluation Purpose |")
    md.append("| :--- | :--- | :--- | :--- |")
    for q in quintiles:
        md.append(f"| **{q['split']}** | `{q['range']}` | ~20% ({q['count']:,}) | {q['purpose']} |")
    md.append("")

    # Local Chunk Analysis
    md.append("## 6. Local (Per-Chunk) Topology Diagnostics\n")
    md.append(f"* **Total Chunk Archives**: `{chunk_summary['total_chunks']}`")
    md.append(f"* **Average Meshes per Chunk**: `{chunk_summary['mean_meshes_per_chunk']:.1f}` (min: `{chunk_summary['min_meshes_per_chunk']}`, max: `{chunk_summary['max_meshes_per_chunk']}`)")
    md.append(f"* **Inter-Chunk Mean Vertex Variance**: `±{chunk_summary['chunk_mean_vertex_std']:.1f}` vertices\n")

    md.append("### Top 5 Densest Chunks (Highest Mean Vertex Count)")
    md.append("| Chunk Archive | Meshes | Mean Vertices | Median Vertices | Max Vertices |")
    md.append("| :--- | :--- | :--- | :--- | :--- |")
    for c in chunk_summary["top_5_densest"]:
        md.append(f"| `{c['chunk_id']}` | {c['count']:,} | {c['mean']:,.1f} | {c['median']:,.1f} | {int(c['max']):,} |")
    md.append("")

    md.append("### Top 5 Lightest Chunks (Lowest Mean Vertex Count)")
    md.append("| Chunk Archive | Meshes | Mean Vertices | Median Vertices | Max Vertices |")
    md.append("| :--- | :--- | :--- | :--- | :--- |")
    for c in chunk_summary["top_5_lightest"]:
        md.append(f"| `{c['chunk_id']}` | {c['count']:,} | {c['mean']:,.1f} | {c['median']:,.1f} | {int(c['max']):,} |")
    md.append("")

    return "\n".join(md)


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive global and local statistical investigation of ABC mesh topology."
    )
    parser.add_argument(
        "-i", "--input", type=str, required=True,
        help="Path to mesh_analysis.json exported from analyze_mesh.py."
    )
    parser.add_argument(
        "-o", "--output", type=str, default="mesh_investigation_report",
        help="Base path/filename for outputs (generates <output>.md and <output>.json, default: 'mesh_investigation_report')."
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Print full summary report to stdout as well."
    )
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        print(f"Error: Input JSON file '{args.input}' does not exist.", file=sys.stderr)
        sys.exit(1)

    print(f"Loading mesh analysis from '{args.input}'...")
    with open(args.input, "r", encoding="utf-8") as f:
        records: List[Dict[str, Any]] = json.load(f)

    total_meshes = len(records)
    print(f"Loaded {total_meshes:,} mesh records. Performing vectorized statistical analysis...")

    # Vectorized array extraction
    vertices = np.array([r["num_vertices"] for r in records], dtype=np.int64)
    faces = np.array([r["num_faces"] for r in records], dtype=np.int64)

    # Global Statistics
    v_stats = compute_array_stats(vertices)
    f_stats = compute_array_stats(faces)

    # Face-to-Vertex Ratio
    valid_mask = vertices > 0
    if np.sum(valid_mask) > 0:
        ratios = faces[valid_mask].astype(np.float64) / vertices[valid_mask].astype(np.float64)
        r_stats = compute_array_stats(ratios)
    else:
        r_stats = {}

    # Semantic Geometric Complexity Tiers
    semantic_bins = [
        (0, 1000, "Tier 1: Primitive / Micro"),
        (1000, 5000, "Tier 2: Low-Poly CAD"),
        (5000, 20000, "Tier 3: Moderate Complexity"),
        (20000, 50000, "Tier 4: Dense CAD"),
        (50000, 100000, "Tier 5: Ultra-Dense"),
        (100000, float("inf"), "Tier 6: Massive / Extreme"),
    ]
    semantic_tiers = compute_binned_distribution(vertices, semantic_bins)

    # Detailed Logarithmic Bins
    log_bins = [
        (0, 100, "< 100"),
        (100, 500, "100 - 500"),
        (500, 1000, "500 - 1K"),
        (1000, 2500, "1K - 2.5K"),
        (2500, 5000, "2.5K - 5K"),
        (5000, 10000, "5K - 10K"),
        (10000, 25000, "10K - 25K"),
        (25000, 50000, "25K - 50K"),
        (50000, 100000, "50K - 100K"),
        (100000, 250000, "100K - 250K"),
        (250000, 500000, "250K - 500K"),
        (500000, float("inf"), ">= 500K"),
    ]
    detailed_histogram = compute_binned_distribution(vertices, log_bins)

    # Recommended Evaluation Quintiles (20% slices)
    q_cuts = np.percentile(vertices, [0, 20, 40, 60, 80, 100])
    quintiles = [
        {
            "split": "Quintile 1 (0-20%)",
            "range": f"[{int(q_cuts[0]):,} — {int(q_cuts[1]):,}]",
            "count": int(np.sum((vertices >= q_cuts[0]) & (vertices <= q_cuts[1]))),
            "purpose": "Primitive / low-poly efficiency & speed evaluation",
        },
        {
            "split": "Quintile 2 (20-40%)",
            "range": f"({int(q_cuts[1]):,} — {int(q_cuts[2]):,}]",
            "count": int(np.sum((vertices > q_cuts[1]) & (vertices <= q_cuts[2]))),
            "purpose": "Standard mechanical parts & baseline geometric evaluation",
        },
        {
            "split": "Quintile 3 (40-60%)",
            "range": f"({int(q_cuts[2]):,} — {int(q_cuts[3]):,}]",
            "count": int(np.sum((vertices > q_cuts[2]) & (vertices <= q_cuts[3]))),
            "purpose": "Medium-complexity CAD assemblies",
        },
        {
            "split": "Quintile 4 (60-80%)",
            "range": f"({int(q_cuts[3]):,} — {int(q_cuts[4]):,}]",
            "count": int(np.sum((vertices > q_cuts[3]) & (vertices <= q_cuts[4]))),
            "purpose": "Dense high-curvature surface reconstruction",
        },
        {
            "split": "Quintile 5 (80-100%)",
            "range": f"({int(q_cuts[4]):,} — {int(q_cuts[5]):,}]",
            "count": int(np.sum((vertices > q_cuts[4]) & (vertices <= q_cuts[5]))),
            "purpose": "Extreme stress-test / scalability benchmark",
        },
    ]

    # Local / Chunk-Level Analysis
    chunks_map: Dict[str, List[int]] = {}
    for r in records:
        parts = r["file_path"].split("/")
        # Expected: 'obj/abc_XXXX_obj_v00/...'
        chunk_id = parts[1] if len(parts) > 1 and parts[0] == "obj" else parts[0]
        chunks_map.setdefault(chunk_id, []).append(r["num_vertices"])

    chunk_stats_list = []
    for c_id, c_verts in chunks_map.items():
        arr_c = np.array(c_verts, dtype=np.int64)
        chunk_stats_list.append({
            "chunk_id": c_id,
            "count": len(arr_c),
            "mean": float(np.mean(arr_c)),
            "median": float(np.median(arr_c)),
            "std": float(np.std(arr_c)),
            "min": float(np.min(arr_c)),
            "max": float(np.max(arr_c)),
        })

    # Sort chunks by mean vertex count
    chunk_stats_sorted = sorted(chunk_stats_list, key=lambda x: x["mean"])
    chunk_counts = [c["count"] for c in chunk_stats_list]
    chunk_means = [c["mean"] for c in chunk_stats_list]

    chunks_summary = {
        "total_chunks": len(chunk_stats_list),
        "mean_meshes_per_chunk": float(np.mean(chunk_counts)),
        "min_meshes_per_chunk": int(np.min(chunk_counts)),
        "max_meshes_per_chunk": int(np.max(chunk_counts)),
        "chunk_mean_vertex_avg": float(np.mean(chunk_means)),
        "chunk_mean_vertex_std": float(np.std(chunk_means)),
        "top_5_lightest": chunk_stats_sorted[:5],
        "top_5_densest": chunk_stats_sorted[-5:][::-1],
        "all_chunks": {c["chunk_id"]: c for c in chunk_stats_list},
    }

    # Assemble complete output structure
    output_data = {
        "dataset_name": "ABC Dataset Mesh Topology",
        "total_meshes": total_meshes,
        "vertices": v_stats,
        "faces": f_stats,
        "face_to_vertex_ratio": r_stats,
        "semantic_tiers": semantic_tiers,
        "detailed_histogram": detailed_histogram,
        "evaluation_quintiles": quintiles,
        "chunks_summary": chunks_summary,
    }

    # Generate Markdown and JSON output paths
    base_out = args.output
    if base_out.endswith(".json") or base_out.endswith(".md"):
        base_out = os.path.splitext(base_out)[0]

    json_path = f"{base_out}.json"
    md_path = f"{base_out}.md"

    # Save outputs
    print(f"Writing structured JSON report to '{json_path}'...")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    print(f"Writing formatted Markdown report to '{md_path}'...")
    md_content = generate_markdown_report(output_data)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    print("\n" + "=" * 70)
    print("Investigation Complete!")
    print(f"  Markdown Report: {os.path.abspath(md_path)}")
    print(f"  JSON Metadata:   {os.path.abspath(json_path)}")
    print("=" * 70)

    if args.verbose:
        print("\n" + md_content)


if __name__ == "__main__":
    main()
