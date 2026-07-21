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

DISTANCES = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 7.5, 10.0]
BATCH     = 128
PARTICLES = 2048
SEED      = 42

# Ordered to surface the proposal axis: each no-proposal baseline sits next to
# its custom-proposal counterpart (MPPI/Biased-MPPI, CEM/Biased-CEM). Proposal
# methods are saturated + solid; no-proposal baselines are light + dashed.
METHODS = ["pfmp_0", "biased_mppi", "mppi", "biased_cem", "cem", "ps", "dial"]
PROPOSAL_METHODS = ("pfmp_0", "biased_mppi", "biased_cem")
LABELS  = {
    "pfmp_0":      "SMC (ours)",
    "biased_mppi": "Biased-MPPI",
    "mppi":        "MPPI",
    "biased_cem":  "Biased-CEM",
    "cem":         "CEM",
    "ps":          "PS",
    "dial":        "DIAL-MPC",
}
COLORS = {
    "pfmp_0":      "#1E8449",  # green (ours)
    "biased_mppi": "#21618C",  # deep blue  — proposal
    "mppi":        "#85C1E9",  # light blue — no proposal
    "biased_cem":  "#B9770E",  # deep amber — proposal
    "cem":         "#F8C471",  # light amber— no proposal
    "ps":          "#E74C3C",  # red
    "dial":        "#9B59B6",  # purple
}
LINE_STYLES = {m: "-" if m in PROPOSAL_METHODS else "--" for m in METHODS}
LINE_WIDTHS = {m: 2.5 if m in PROPOSAL_METHODS else 1.8 for m in METHODS}

OUTPUT_DIR = Path(__file__).parent.parent / "figures"
DATA_PATH  = OUTPUT_DIR / "fig_pushT_data.json"


def run_experiment(distances=None, batch=BATCH, particles=PARTICLES, seed=SEED, tuned_params_dir=None):
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
        cmd = [
            sys.executable, "examples/2d_pushT.py",
            "--distance",  str(d),
            "--batch",     str(batch),
            "--particles", str(particles),
            "--seed",      str(seed),
            "--out",       str(cache),
        ]
        if tuned_params_dir is not None:
            cmd += ["--tuned-params-dir", str(tuned_params_dir)]
        subprocess.run(cmd, check=True)
        with open(cache) as f:
            results[key] = json.load(f)

    with open(DATA_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nAll results saved to {DATA_PATH}")
    return results


def _wilson_ci(k, n, z=1.96):
    """Wilson score interval (in %) for k successes out of n Bernoulli trials.
    Unlike the normal approximation, it stays well-defined and non-zero-width
    at p=0 and p=1, the regime our success curves live in."""
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return 100.0 * (center - half), 100.0 * (center + half)


def make_plots(results):
    dists = sorted(float(k) for k in results)
    n_obs_all = [max(0, int(d) - 3) for d in dists]
    seen, keep = set(), []
    for i, n in enumerate(n_obs_all):
        if n not in seen:
            seen.add(n); keep.append(i)
    dists = [dists[i] for i in keep]
    x = np.array([n_obs_all[i] for i in keep])

    success_rates, ci_lo, ci_hi = {}, {}, {}
    for m in METHODS:
        rates, los, his = [], [], []
        for d in dists:
            data = results[str(d)]
            v = data.get(m)
            if v is None:
                rates.append(float("nan")); los.append(float("nan")); his.append(float("nan"))
                continue
            k, n = (sum(v), len(v)) if isinstance(v, list) else (v, data.get("batch", BATCH))
            rates.append(100.0 * k / n if n > 0 else float("nan"))
            lo, hi = _wilson_ci(k, n)
            los.append(lo); his.append(hi)
        success_rates[m] = np.array(rates)
        ci_lo[m] = np.array(los); ci_hi[m] = np.array(his)

    fig, ax = plt.subplots(figsize=(5, 4.5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5, color="gray")

    for m in METHODS:
        ax.fill_between(x, ci_lo[m], ci_hi[m], color=COLORS[m], alpha=0.15, linewidth=0,
                        zorder=4 if m.startswith("pfmp") else 2)
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
    parser.add_argument("--tuned-params-dir", type=str, default=None,
                        help="Directory with pushT_optuna_{method}.json files for tuned baselines.")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Override output directory (default: figures/).")
    parser.add_argument("--plot-ablation", action="store_true",
                        help="Plot particle-count ablation from --ablation-dirs.")
    parser.add_argument("--ablation-dirs", type=str, nargs="+", default=None,
                        metavar="DIR",
                        help="Directories figures/ablation_N{p}/ in ascending particle order.")
    parser.add_argument("--ablation-particles", type=int, nargs="+", default=None,
                        metavar="N",
                        help="Particle counts matching --ablation-dirs (e.g. 1 16 128 512 2048).")
    parser.add_argument("--ablation-distance", type=float, default=4.0,
                        help="Distance used for the ablation (to load the right JSON).")
    args = parser.parse_args()

    if args.output_dir is not None:
        OUTPUT_DIR = Path(args.output_dir)
        DATA_PATH  = OUTPUT_DIR / "fig_pushT_data.json"

    if args.plot_ablation:
        if not args.ablation_dirs or not args.ablation_particles:
            parser.error("--plot-ablation requires --ablation-dirs and --ablation-particles")
        d_key = str(float(args.ablation_distance))
        fname = f"pushT_d{args.ablation_distance:.1f}.json"
        rates = {m: [] for m in METHODS}
        ci_lo = {m: [] for m in METHODS}
        ci_hi = {m: [] for m in METHODS}
        for d in args.ablation_dirs:
            p = Path(d) / fname
            if not p.exists():
                raise FileNotFoundError(f"Missing {p}")
            with open(p) as f:
                data = json.load(f)
            for m in METHODS:
                v = data.get(m)
                if v is None:
                    rates[m].append(float("nan")); ci_lo[m].append(float("nan")); ci_hi[m].append(float("nan"))
                    continue
                k, n = (sum(v), len(v)) if isinstance(v, list) else (v, data.get("batch", BATCH))
                rates[m].append(100.0 * k / n if n > 0 else float("nan"))
                lo, hi = _wilson_ci(k, n)
                ci_lo[m].append(lo); ci_hi[m].append(hi)

        x = np.array(args.ablation_particles, dtype=float)

        fig, ax = plt.subplots(figsize=(5, 4.5))
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")
        ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5, color="gray")
        for m in METHODS:
            ax.fill_between(x, ci_lo[m], ci_hi[m], color=COLORS[m], alpha=0.15, linewidth=0,
                            zorder=4 if m == "pfmp_0" else 2)
            ax.plot(x, rates[m],
                    color=COLORS[m], linestyle=LINE_STYLES[m],
                    linewidth=LINE_WIDTHS[m], marker="o", markersize=5,
                    label=LABELS[m], zorder=5 if m == "pfmp_0" else 3)
        ax.set_xscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels([str(p) for p in args.ablation_particles], fontsize=11)
        ax.set_xlabel("Particles (log scale)", fontsize=13)
        ax.set_ylim(-5, 105)
        ax.tick_params(labelsize=11)
        ax.legend(fontsize=10, framealpha=0.95, edgecolor="black", fancybox=False)
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)
        fig.tight_layout()
        out = OUTPUT_DIR / "fig_pushT_ablation"
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(out) + ".pdf", dpi=300, bbox_inches="tight")
        fig.savefig(str(out) + ".png", dpi=300, bbox_inches="tight")
        print(f"Saved {out}.pdf / .png")
        plt.show()

    elif args.plot_only:
        results = {}
        if DATA_PATH.exists():
            with open(DATA_PATH) as f:
                results = json.load(f)
        # Merge per-distance cache files (for distances in DISTANCES) not yet
        # in the main data file. Stale caches for other distances are ignored.
        valid = {str(d) for d in DISTANCES}
        for cache in sorted(OUTPUT_DIR.glob("pushT_d*.json")):
            d = str(float(cache.stem.replace("pushT_d", "").replace("_", ".")))
            if d in valid and d not in results:
                with open(cache) as f:
                    results[d] = json.load(f)
        results = {d: v for d, v in results.items() if d in valid}
        if not results:
            raise FileNotFoundError("No results found — run --experiment first.")
        make_plots(results)
    elif args.experiment:
        results = run_experiment(args.distances, args.batch, args.particles, args.seed,
                                 tuned_params_dir=args.tuned_params_dir)
        make_plots(results)
    else:
        parser.print_help()
