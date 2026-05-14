"""
2D push-T distance sweep — success rate vs. problem difficulty.

Mirrors the structure of 2d_geometric.py: run with --experiment to collect
data (caches per-distance JSON), then --plot-only to regenerate figures.

Usage:
    python experiments/pushT_2d_sweep.py --experiment
    python experiments/pushT_2d_sweep.py --experiment --distances 4 5 6
    python experiments/pushT_2d_sweep.py --plot-only
"""

import json
import subprocess
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

DISTANCES = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 7.5, 10.0, 15.0, 20.0]
BATCH     = 128
PARTICLES = 2048
SEED      = 42

METHODS = ["pfmp_0", "pfmp_025", "pfmp_05", "ps", "mppi", "cem", "dial"]
LABELS  = {
    "pfmp_0":   "PFMP (no resamp.)",
    "pfmp_025": "PFMP (ESS=0.25)",
    "pfmp_05":  "PFMP (ESS=0.5)",
    "ps":       "PS",
    "mppi":     "MPPI",
    "cem":      "CEM",
    "dial":     "DIAL",
}
COLORS = {
    "pfmp_0":   "#27AE60",
    "pfmp_025": "#1ABC9C",
    "pfmp_05":  "#82E0AA",
    "ps":       "#E74C3C",
    "mppi":     "#3498DB",
    "cem":      "#E67E22",
    "dial":     "#9B59B6",
}
LINE_STYLES = {m: "-" if m.startswith("pfmp") else "--" for m in METHODS}
LINE_WIDTHS = {m: 2.5 if m.startswith("pfmp") else 1.8 for m in METHODS}

OUTPUT_DIR = Path(__file__).parent.parent / "figures"
DATA_PATH  = OUTPUT_DIR / "fig_pushT_data.json"


def run_experiment(distances=None, batch=BATCH, particles=PARTICLES, seed=SEED):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dists = distances if distances is not None else DISTANCES
    results = {}

    # Load existing partial results so we can skip completed distances.
    if DATA_PATH.exists():
        with open(DATA_PATH) as f:
            results = json.load(f)

    for d in dists:
        key = str(d)
        cache = OUTPUT_DIR / f"pushT_d{d:.1f}.json"

        if cache.exists():
            print(f"[cached] distance={d:.1f}  ({cache})")
            with open(cache) as f:
                results[key] = json.load(f)
            continue

        print(f"\n{'='*60}")
        print(f"  distance={d:.1f}  batch={batch}  particles={particles}")
        print(f"{'='*60}")
        subprocess.run(
            [
                sys.executable, "examples/2d_pushT.py",
                "--distance",  str(d),
                "--batch",     str(batch),
                "--particles", str(particles),
                "--seed",      str(seed),
                "--out",       str(cache),
            ],
            check=True,
        )
        with open(cache) as f:
            results[key] = json.load(f)

    with open(DATA_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nAll results saved to {DATA_PATH}")
    return results


def make_plots(results):
    dists = sorted(float(k) for k in results)
    n_obs_all = [max(0, int(d) - 3) for d in dists]
    seen, keep = set(), []
    for i, n in enumerate(n_obs_all):
        if n not in seen:
            seen.add(n); keep.append(i)
    dists = [dists[i] for i in keep]
    x = np.array([n_obs_all[i] for i in keep])

    success_rates = {}
    for m in METHODS:
        rates = []
        for d in dists:
            data = results[str(d)]
            v = data.get(m)
            if v is None:
                rates.append(float("nan"))
            elif isinstance(v, list):
                rates.append(100.0 * sum(v) / len(v) if v else float("nan"))
            else:
                n = data.get("batch", BATCH)
                rates.append(100.0 * v / n if n > 0 else float("nan"))
        success_rates[m] = np.array(rates)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5, color="gray")

    for m in METHODS:
        ax.plot(
            x, success_rates[m],
            color=COLORS[m], linestyle=LINE_STYLES[m],
            linewidth=LINE_WIDTHS[m], marker="o", markersize=5,
            label=LABELS[m], zorder=5 if m.startswith("pfmp") else 3,
        )

    ax.set_xlabel("Number of obstacles", fontsize=13)
    ax.set_ylabel("Success rate (%)", fontsize=13)
    ax.set_xticks(x)
    ax.set_ylim(-5, 105)
    ax.axhline(100, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
    ax.tick_params(labelsize=11)
    ax.legend(fontsize=10, framealpha=0.95, edgecolor="black", fancybox=False,
              loc="upper right")
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
    ax.set_title("Simulated Push-T (Success Rate vs. Obstacles)", fontsize=13, pad=10)
    fig.tight_layout()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / "fig_pushT.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / "fig_pushT.png", dpi=300, bbox_inches="tight")
    print("Saved fig_pushT.pdf / fig_pushT.png")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Distance sweep for 2D push-T benchmark."
    )
    parser.add_argument("--experiment", action="store_true",
                        help="Run sweep and save results.")
    parser.add_argument("--plot-only", action="store_true",
                        help="Re-plot from saved data (no re-running).")
    parser.add_argument("--distances", type=float, nargs="+", default=None,
                        help="Subset of distances to run (default: all).")
    parser.add_argument("--batch",     type=int,   default=BATCH)
    parser.add_argument("--particles", type=int,   default=PARTICLES)
    parser.add_argument("--seed",      type=int,   default=SEED)
    args = parser.parse_args()

    if args.plot_only:
        results = {}
        if DATA_PATH.exists():
            with open(DATA_PATH) as f:
                results = json.load(f)
        # Merge any per-distance cache files not yet in the main data file.
        for cache in sorted(OUTPUT_DIR.glob("pushT_d*.json")):
            d = str(float(cache.stem.replace("pushT_d", "").replace("_", ".")))
            if d not in results:
                with open(cache) as f:
                    results[d] = json.load(f)
        if not results:
            raise FileNotFoundError("No results found — run --experiment first.")
        make_plots(results)
    elif args.experiment:
        results = run_experiment(args.distances, args.batch, args.particles, args.seed)
        make_plots(results)
    else:
        parser.print_help()
