"""Sweeps and figures for the tasks in `geom_runner.py`.

Two roles, both of which used to live in a separate script per task
(`pushT_2d_sweep.make_plots`, `drive.make_plots`, `mbm_figure`):

Turns a sweep's results into the paper figure. The axis itself is defined in
the task's config (`sweep:` block) and run by `geom_runner`.

Per-problem values are cached raw, never pre-aggregated: a success rate needs
its numerator and denominator to get a Wilson interval, and a cost needs the
whole sample to get a median and IQR. Re-plotting never re-runs.

Takes a results path and nothing else: `geom_runner` writes the axis and the
method list into the cache beside the raw values, so a figure never depends on
a yaml that may have moved on since the run.

Usage:
    python experiments/geom_runner.py  experiments/configs/pushT.yaml out/pushT
    python experiments/geom_analyze.py out/pushT
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Callable, NamedTuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))



from pfmp import method_decomp

OUTPUT_DIR = Path(__file__).parent.parent / "figures"

# Shared with pushT_2d_sweep so figures stay comparable across the paper. Solid
# + heavy = has an object-informed proposal; dashed + light = does not.
LABELS = {
    # SIS/SIR are the same update (argmin) at two resampling thresholds, so
    # they are labelled by rho rather than by two names; `label_for` fills the
    # value in from the run. `ps` is that same algorithm on a task with no
    # custom proposal.
    "sis": "PS", "ps": "PS", "sir": "PS",
    "biased_mppi": "Biased-MPPI", "mppi": "MPPI",
    "biased_cem": "Biased-CEM", "cem": "CEM",
    "sis_sample": "SIS+sample", "dial": "DIAL-MPC",
}
COLORS = {
    "sis": "#E74C3C", "ps": "#E74C3C",   # red: the argmin update
    "sir": "#1E8449",                    # green: resampling
    "biased_mppi": "#21618C", "mppi": "#85C1E9",
    "biased_cem": "#B9770E", "cem": "#F8C471",
    "sis_sample": "#52BE80", "dial": "#9B59B6",
}
def label_for(alg, hyperparams, style="rho"):
    """Method label carrying the resampling threshold it actually ran at.

    Resampling is one of the three axes, so every method states its rho rather
    than hiding it in a name: `sis` and `sir` become PS at rho 0 and rho 0.5,
    and `mppi+r` is MPPI at the same rho as SIR.
    """
    proposal, update, rho = method_decomp(alg)
    if style == "proposal":
        return f"{UPDATE_LABEL.get(update, update)}, {PROPOSAL_LABEL[proposal]}"
    if rho is None:      # bare "+r": the run recorded what it resolved to
        rho = (hyperparams or {}).get(alg, {}).get("ess_thresh", 0.0)
    return rf"{UPDATE_LABEL.get(update, update)} ($\rho$ = {rho:g})"


# Line style carries rho, hue carries the update. Assigned in sorted rho order,
# so more resampling reads as a more solid line: dotted (none) -> dashed -> solid.
STYLE_CYCLE = (":", "--", "-", (0, (3, 1, 1, 1)), (0, (5, 1)), (0, (1, 1)))


# `pfmp.method_decomp` is the single decoder; these only name the levels.
UPDATE_LABEL = {"argmin": "argmin (PS)", "sample": "sample", "mean": "MPPI",
                "topk": "CEM", "diffuse": "DIAL-MPC"}
PROPOSAL_LABEL = {"informed": "object-informed", "gaussian": "Gaussian",
                  "learned": "learned"}


def proposal_of(alg):
    """(update, proposal) -- the two categorical axes."""
    proposal, update, _ = method_decomp(alg)
    return update, proposal


def rho_of(alg, hyperparams):
    """(update, rho), with a bare `+r` resolved from what the run recorded."""
    _, update, rho = method_decomp(alg)
    if rho is None:
        rho = (hyperparams or {}).get(alg, {}).get("ess_thresh", 0.0)
    return update, float(rho)


def encodings(algorithms, hyperparams, style="rho"):
    """{group: colour}, {style key: linestyle}. `style` selects which axis the
    line style carries: the resampling threshold, or the proposal."""
    groups, keys = [], []
    for alg in algorithms:
        g, k = _axes_of(alg, hyperparams, style)
        if g not in groups:
            groups.append(g)
        if k not in keys:
            keys.append(k)
    colour = {g: _colour(g) for g in groups}
    order = sorted(keys) if style == "rho" else keys
    return colour, {k: STYLE_CYCLE[i % len(STYLE_CYCLE)]
                    for i, k in enumerate(order)}


def _axes_of(alg, hyperparams, style):
    """(hue group, line-style key) for one method under the chosen encoding."""
    if style == "proposal":
        return proposal_of(alg)
    return rho_of(alg, hyperparams)


# Hue follows the update; the other two axes are carried by line style and by
# faceting, so a resampling variant or a different proposal keeps its colour.
UPDATE_COLORS = {"argmin": "#E74C3C", "sample": "#52BE80", "mean": "#85C1E9",
                 "topk": "#F8C471", "diffuse": "#9B59B6"}


def _colour(key):
    if key in UPDATE_COLORS:
        return UPDATE_COLORS[key]
    try:
        return UPDATE_COLORS.get(method_decomp(key)[1], "gray")
    except ValueError:
        return "gray"


# ---------------------------------------------------------------------------
# Sweep definitions
# ---------------------------------------------------------------------------

class Sweep(NamedTuple):
    """The plotted axis, read straight off the config's `sweep` block. `xticks`
    is the coordinate each value is drawn at, which is not the value itself when
    the knob and the reported quantity differ (push-T sweeps arena size but
    reports obstacle count) or when the axis is categorical."""
    key: str
    values: list
    xticks: list                   # parallel to values; None = use the values
    xlabel: str
    panels: list                   # [{metric, kind, ylabel}] one row each;
                                   # kind "rate" -> Wilson CI, else median+IQR
    facet: str                     # "update" -> one column per update rule
    style: str                     # what the line style encodes: "rho"/"proposal"
    name: str

    def x(self, i):
        return self.xticks[i] if self.xticks else self.values[i]


def load_sweep(blk):
    """Build the Sweep from the axis the cache stored beside its results.

    `panels` is the general form; a bare metric/kind/ylabel is read as a single
    panel so a one-plot sweep needs no extra nesting in the yaml.
    """
    panels = blk.get("panels") or [{"metric": blk.get("metric", "success"),
                                    "kind": blk.get("kind", "rate"),
                                    "ylabel": blk.get("ylabel", "")}]
    return Sweep(key=blk["key"], values=blk["values"], xticks=blk.get("xticks"),
                 xlabel=blk.get("xlabel", blk["key"]), panels=panels,
                 facet=blk.get("facet"), style=blk.get("style", "rho"),
                 name=blk.get("name", f"{blk['key']}_fig"))


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def wilson_ci(k, n, z=1.96):
    """Wilson score interval, in percent. Beats the normal approximation at the
    0% and 100% ends, which is exactly where these success rates sit."""
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return 100 * (centre - half), 100 * (centre + half)


def aggregate(values, kind):
    """(centre, lo, hi) for one cell, by metric kind."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return (float("nan"),) * 3
    if kind == "rate":
        k, n = int(v.sum()), v.size
        lo, hi = wilson_ci(k, n)
        return 100.0 * k / n, lo, hi
    return float(np.median(v)), float(np.percentile(v, 25)), float(np.percentile(v, 75))


def series(results, sweep, algorithms, metric, kind):
    """{algorithm: (x, centre, lo, hi)}, deduped on the plotted x coordinate."""
    xs, keys = [], []
    for i, value in enumerate(sweep.values):
        if str(value) not in results:
            continue
        xv = sweep.x(i)
        if xv in xs:            # several arena sizes share an obstacle count
            continue
        xs.append(xv)
        keys.append(str(value))

    out = {}
    for alg in algorithms:
        c, lo, hi = [], [], []
        for k in keys:
            cell = results[k].get(alg)
            trio = ((float("nan"),) * 3 if cell is None
                    else aggregate(cell[metric], kind))
            c.append(trio[0]); lo.append(trio[1]); hi.append(trio[2])
        out[alg] = (xs, np.array(c), np.array(lo), np.array(hi))
    return out


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _draw_panel(ax, data, algorithms, sweep, panel, categorical, legend, hp):
    import numpy as np
    ax.set_facecolor("white")
    ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5, color="gray",
            axis="y" if categorical else "both")
    ax.set_axisbelow(True)

    if categorical:
        labels = data[algorithms[0]][0]
        idx = np.arange(len(labels))
        width = 0.8 / len(algorithms)
        for i, alg in enumerate(algorithms):
            _, c, lo, hi = data[alg]
            ax.bar(idx + i * width - 0.4 + width / 2, c, width * 0.92,
                   color=_colour(alg), label=label_for(alg, hp),
                   yerr=[c - lo, hi - c], capsize=2,
                   error_kw=dict(lw=0.8, ecolor="0.3"))
        ax.set_xticks(idx)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    else:
        colours, styles = encodings(algorithms, hp, sweep.style)
        for alg in algorithms:
            x, c, lo, hi = data[alg]
            grp, skey = _axes_of(alg, hp, sweep.style)
            ax.fill_between(x, lo, hi, color=colours[grp],
                            alpha=0.10, linewidth=0, zorder=2)
            ax.plot(x, c, color=colours[grp], linestyle=styles[skey],
                    linewidth=2.0, marker="o", markersize=4, zorder=4,
                    label="_nolegend_")
        ax.set_xticks(data[algorithms[0]][0])
        if panel["kind"] == "rate":
            ax.set_ylim(-5, 105)
            ax.axhline(100, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)

    ax.set_xlabel(sweep.xlabel, fontsize=12)
    ax.set_ylabel(panel.get("ylabel") or panel["metric"], fontsize=12)
    ax.tick_params(labelsize=10)
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)


def _split_legend(fig, algorithms, hp, faceted, style="rho"):
    """One legend strip above the grid: hue = update rule, line style = rho.

    Figure-level and outside the axes, because `loc="best"` has no good answer
    on a faceted grid: every panel is full of data and matplotlib will drop the
    box on top of a curve. When faceted, the column titles already name the
    update, so only the rho key is shown.
    """
    from matplotlib.lines import Line2D

    colours, styles = encodings(algorithms, hp, style)
    fmt = ((lambda k: PROPOSAL_LABEL[k]) if style == "proposal"
           else (lambda k: rf"$\rho$ = {k:g}"))
    handles = [Line2D([], [], color="0.35", lw=2.0, linestyle=st, label=fmt(k))
               for k, st in styles.items()]
    if not faceted and len(colours) > 1:
        handles = [Line2D([], [], color=c, lw=2.2,
                          label=UPDATE_LABEL.get(g, LABELS.get(g, g)))
                   for g, c in colours.items()] + handles
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0),
               ncol=len(handles), fontsize=10, framealpha=0.95,
               edgecolor="black", fancybox=False)


def make_figure(results, sweep, algorithms, output_dir=OUTPUT_DIR, hp=None):
    """Rows are metrics; columns are facets. Faceting by update rule keeps each
    cell to one hue and a few markers: nine series with nine confidence bands in
    one axes is unreadable, and curves that coincide (SIS under MPPI, on a task
    where they are the same algorithm) vanish entirely."""
    import matplotlib.pyplot as plt

    if sweep.facet == "update":
        seen = {}
        for alg in algorithms:
            seen.setdefault(_axes_of(alg, hp, sweep.style)[0], []).append(alg)
        groups = [(UPDATE_LABEL.get(g, LABELS.get(g, g)), algs)
                  for g, algs in seen.items()]
    else:
        groups = [(None, list(algorithms))]

    probe = series(results, sweep, algorithms, sweep.panels[0]["metric"],
                   sweep.panels[0]["kind"])
    categorical = isinstance(next(iter(probe.values()))[0][0], str)
    nrow, ncol = len(sweep.panels), len(groups)

    fig, axes = plt.subplots(nrow, ncol, squeeze=False, sharex=True,
                             sharey="row" if ncol > 1 else False,
                             figsize=((6.5 if categorical else 4.2) * ncol,
                                      4.2 * nrow))
    fig.patch.set_facecolor("white")

    for r, panel in enumerate(sweep.panels):
        for c, (title, algs) in enumerate(groups):
            ax = axes[r][c]
            data = series(results, sweep, algs, panel["metric"], panel["kind"])
            _draw_panel(ax, data, algs, sweep, panel, categorical,
                        legend=False, hp=hp)
            if title and r == 0:
                ax.set_title(title, fontsize=12)
            if c > 0:
                ax.set_ylabel("")
            if r < nrow - 1:
                ax.set_xlabel("")
    fig.tight_layout()
    if not categorical:
        _split_legend(fig, algorithms, hp, faceted=(sweep.facet is not None),
                      style=sweep.style)

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / sweep.name
    for ext in ("pdf", "png"):
        fig.savefig(f"{stem}.{ext}", dpi=300, bbox_inches="tight")
    print(f"Saved {stem}.pdf / {stem}.png")
    return fig


def print_table(results, sweep, algorithms, hp=None):
    for panel in sweep.panels:
        data = series(results, sweep, algorithms, panel["metric"], panel["kind"])
        xs = data[algorithms[0]][0]
        print(f"\n{panel['metric']} ({panel['kind']})")
        print(f"{'method':<14}" + "".join(f"{str(x):>12}" for x in xs))
        for alg in algorithms:
            _, c, _, _ = data[alg]
            print(f"{label_for(alg, hp, sweep.style):<26}" + "".join(f"{v:>12.3f}" for v in c))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results", help="sweep_<key>.json, or a dir holding one")
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--algorithms", nargs="+", default=None,
                    help="subset/order to plot (default: what the sweep ran)")
    args = ap.parse_args()

    path = Path(args.results)
    if path.is_dir():
        found = sorted(path.glob("sweep_*.json"))
        if len(found) != 1:
            raise SystemExit(f"expected one sweep_*.json in {path}, found {found}")
        path = found[0]
    blob = json.load(open(path))

    sweep = load_sweep(blob["sweep"])
    algorithms = args.algorithms or blob["algorithms"]
    results = blob["results"]

    hp = blob.get("hyperparams")
    print_table(results, sweep, algorithms, hp)
    make_figure(results, sweep, algorithms,
                Path(args.output_dir) if args.output_dir else OUTPUT_DIR, hp)


if __name__ == "__main__":
    main()
