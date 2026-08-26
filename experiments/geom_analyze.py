"""Figures for the sweeps in `geom_runner.py`.

    python experiments/geom_runner.py  experiments/configs/push_tee.yaml out/push_tee
    python experiments/geom_analyze.py out/push_tee
"""

import argparse
import json
import sys
from pathlib import Path
from typing import NamedTuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))


from pfmp import method_decomp
from geom_runner import ESS_THRESH_DEFAULT

OUTPUT_DIR = Path(__file__).parent.parent / "figures"

LABELS = {
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
    "sis_sample": "#52BE80", "dial": "#4A235A",
}
def label_for(alg, hyperparams, style="rho"):
    """Method label stating the rho it actually ran at."""
    proposal, update, rho = method_decomp(alg)
    if style == "proposal":
        return f"{UPDATE_LABEL.get(update, update)}, {PROPOSAL_LABEL[proposal]}"
    name = AXIS_LABEL.get((proposal, update), f"{proposal}/{update}")
    if rho is None:
        # A bare `+r` means "the task's threshold", which is pinned globally;
        # older caches predate `used_hp` recording it.
        rho = (hyperparams or {}).get(alg, {}).get("ess_thresh", ESS_THRESH_DEFAULT)
    # "+R" marks the resampling variant, so the name states the rho axis and the
    # parenthetical states its value.
    return rf"{name}{'+R' if rho else ''} ($\rho$ = {rho:.3g})"


# Ordered by rho: dotted for rho = 0 (no resampling), solid for the
# resampling variant. Matches the plan figure, where the SIS panel is
# dotted and SIS/R solid.
STYLE_CYCLE = (":", "-", "--", (0, (3, 1, 1, 1)), (0, (5, 1)), (0, (1, 1)))

AXIS_LABEL = {
    ("gaussian", "argmin"):  "PS",
    ("informed", "argmin"):  "Biased-PS",
    ("learned",  "argmin"):  "learned",
    ("gaussian", "mean"):    "MPPI",
    ("informed", "mean"):    "Biased-MPPI",
    ("gaussian", "topk"):    "CEM",
    ("informed", "topk"):    "Biased-CEM",
    ("gaussian", "diffuse"): "DIAL-MPC",
    ("informed", "sample"):  "SIS-sample",
    ("learned",  "sample"):  "Piche",
}

# `pfmp.method_decomp` is the single decoder; these only name the levels.
UPDATE_LABEL = {"argmin": "PS", "sample": "sample", "mean": "MPPI",
                "topk": "CEM", "diffuse": "DIAL-MPC"}
PROPOSAL_LABEL = {"informed": "Programmable", "gaussian": "Gaussian",
                  "learned": "Learned"}


def proposal_of(alg):
    """(update, proposal) -- the two categorical axes."""
    proposal, update, _ = method_decomp(alg)
    return update, proposal


def rho_of(alg, hyperparams):
    """((proposal, update), rho), with a bare `+r` resolved from the run.

    Hue keys on the whole cell so `mppi` and `biased_mppi` do not collide.
    """
    proposal, update, rho = method_decomp(alg)
    if rho is None:
        rho = (hyperparams or {}).get(alg, {}).get("ess_thresh", 0.0)
    return (proposal, update), float(rho)


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


def _group_label(g):
    """Legend/facet name for a hue group: a (proposal, update) cell, or a bare
    update when the proposal view has already split them into columns."""
    if isinstance(g, tuple):
        return AXIS_LABEL.get(g, "/".join(g))
    return UPDATE_LABEL.get(g, LABELS.get(g, g))


def _axes_of(alg, hyperparams, style):
    """(hue group, line-style key) for one method under the chosen encoding."""
    if style == "proposal":
        return proposal_of(alg)
    return rho_of(alg, hyperparams)


AXIS_COLORS = {
    ("gaussian", "argmin"):  "#1E8449",   # PS           green
    ("informed", "argmin"):  "#145A32",   # Biased-PS    deep green (ours)
    ("gaussian", "mean"):    "#2E86C1",   # MPPI         blue (= UPDATE_COLORS)
    ("informed", "mean"):    "#21618C",   # Biased-MPPI  deep blue
    ("gaussian", "topk"):    "#E67E22",   # CEM          amber (= UPDATE_COLORS)
    ("informed", "topk"):    "#B9770E",   # Biased-CEM   deep amber
    ("gaussian", "diffuse"): "#4A235A",   # DIAL-MPC     purple
    ("informed", "sample"):  "#52BE80",   # SIS-sample   light green
    ("learned",  "sample"):  "#7F8C8D",   # Piche        grey
    ("learned",  "argmin"):  "#95A5A6",
}

UPDATE_COLORS = {"argmin": "#1E8449", "mean": "#2E86C1", "topk": "#E67E22",
                 "diffuse": "#4A235A", "sample": "#7F8C8D"}

BAR_FILL = {
    ("gaussian", False): ("///", "Gaussian"),
    ("informed", False): ("",    "Programmable"),
    ("informed", True):  (".",   "Programmable + Resampling"),
}


def _bar_colour_for(update):
    return UPDATE_COLORS.get(update, "gray")


def _bar_fill(alg):
    """(hatch, label) for the merged proposal/rho channel. See `BAR_FILL`."""
    proposal, _, rho = method_decomp(alg)
    resampling = rho is None or rho > 0
    if (proposal, resampling) not in BAR_FILL:
        raise ValueError(
            f"{alg}: no bar fill for {(proposal, resampling)}. The fill "
            "collapses proposal and rho into one ladder, which only works "
            "while no Gaussian method resamples; give it its own channel.")
    return BAR_FILL[(proposal, resampling)]


def _line_marker(alg):
    """Filled marker = biased (informed) proposal, hollow = unbiased.

    Lines have only hue and dash, which is one channel short of the three axes,
    so the marker's fill carries the proposal -- the same job the hatch does on
    the bars.
    """
    try:
        informed = method_decomp(alg)[0] == "informed"
    except ValueError:
        informed = False
    return "o", (None if informed else "white")


def _bar_colour(alg):
    try:
        return UPDATE_COLORS.get(method_decomp(alg)[1], "gray")
    except ValueError:
        return "gray"


def _hatch(alg):
    return _bar_fill(alg)[0]


def _colour(key):
    if isinstance(key, tuple):
        return AXIS_COLORS.get(key, "gray")
    try:
        proposal, update, _ = method_decomp(key)
    except ValueError:
        return "gray"
    return AXIS_COLORS.get((proposal, update), "gray")


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
    xscale: str                    # "linear" | "log"; log for particle counts,
                                   #   where the values are geometric
    facet: str                     # "update" -> one column per update rule
    style: str                     # what the line style encodes: "rho"/"proposal"
    name: str

    def x(self, i):
        return self.xticks[i] if self.xticks else self.values[i]


def load_sweep(blk):
    """Build the Sweep from the axis the cache stored beside its results."""
    panels = blk.get("panels") or [{"metric": blk.get("metric", "success"),
                                    "kind": blk.get("kind", "rate"),
                                    "ylabel": blk.get("ylabel", "")}]
    return Sweep(key=blk["key"], values=blk["values"], xticks=blk.get("xticks"),
                 xlabel=blk.get("xlabel", blk["key"]), panels=panels,
                 xscale=blk.get("xscale", "linear"),
                 facet=(None if blk.get("facet") in (None, "none", "None")
                        else blk["facet"]),
                 style=blk.get("style", "rho"),
                 name=blk.get("name", f"{blk['key']}_fig"))


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def wilson_ci(k, n, z=1.96):
    """Wilson score interval, in percent for success rate."""
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

def _draw_panel(ax, data, algorithms, sweep, panel, categorical, legend, hp,
                style_algs=None):
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
                   color=_bar_colour(alg), label="_nolegend_",
                   hatch=_hatch(alg), edgecolor="white", linewidth=0.0,
                   yerr=[np.maximum(c - lo, 0.0), np.maximum(hi - c, 0.0)],
                   capsize=2,
                   error_kw=dict(lw=0.8, ecolor="0.3"))
        ax.set_xticks(idx)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=11)
    else:
        _, styles = encodings(style_algs or algorithms, hp, sweep.style)
        for alg in algorithms:
            x, c, lo, hi = data[alg]
            _, skey = _axes_of(alg, hp, sweep.style)
            col = _bar_colour(alg)
            mk, mfc = _line_marker(alg)
            ax.fill_between(x, lo, hi, color=col,
                            alpha=0.10, linewidth=0, zorder=2)
            ax.plot(x, c, color=col, linestyle=styles[skey],
                    linewidth=2.0, marker=mk, markersize=5,
                    markerfacecolor=(col if mfc is None else mfc),
                    markeredgecolor=col, markeredgewidth=1.4,
                    zorder=4, label="_nolegend_")
        xs = data[algorithms[0]][0]
        if sweep.xscale == "log":
            ax.set_xscale("log")
            ax.set_xticks(xs)
            ax.set_xticklabels([f"{x:g}" for x in xs])
            ax.minorticks_off()
        else:
            ax.set_xticks(xs)
        if panel["kind"] == "rate":
            ax.set_ylim(-5, 105)
            ax.axhline(100, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)

    ax.set_xlabel(sweep.xlabel, fontsize=14)
    ax.set_ylabel(panel.get("ylabel") or panel["metric"], fontsize=14)
    ax.tick_params(labelsize=12)
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)


def _legend_rows(fig, rows):
    from matplotlib.lines import Line2D

    rows = [r for r in rows if r]
    if not rows:
        return
    ncol = max(len(r) for r in rows)
    padded = [r + [Line2D([], [], color="none", label=" ")] * (ncol - len(r))
              for r in rows]
    handles = [padded[r][c] for c in range(ncol) for r in range(len(padded))]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0),
               ncol=ncol, fontsize=11, framealpha=0.95,
               edgecolor="black", fancybox=False)


def _split_legend(fig, algorithms, hp, faceted, style="rho"):
    from matplotlib.lines import Line2D

    _, styles = encodings(algorithms, hp, style)
    fmt = ((lambda k: PROPOSAL_LABEL[k]) if style == "proposal"
           else (lambda k: rf"$\rho$ = {k:.3g}"))
    handles, proposal_handles = [], []
    if not faceted:
        keys = []
        for a in algorithms:
            u = method_decomp(a)[1]
            if u not in keys:
                keys.append(u)
        handles = [Line2D([], [], color=_bar_colour_for(u), lw=2.2,
                          label=UPDATE_LABEL.get(u, u)) for u in keys]
        if len({proposal_of(a)[1] for a in algorithms}) > 1:
            proposal_handles = [
                Line2D([], [], color="0.35", lw=0, marker="o", markersize=6,
                       markerfacecolor="0.35", label="Programmable"),
                Line2D([], [], color="0.35", lw=0, marker="o", markersize=6,
                       markerfacecolor="white", markeredgewidth=1.4,
                       label="Gaussian")]
    rho_handles = [Line2D([], [], color="0.35", lw=2.0, linestyle=st, label=fmt(k))
                   for k, st in styles.items()]
    _legend_rows(fig, [handles, proposal_handles + rho_handles])


def _bar_legend(fig, algorithms, hp):
    from matplotlib.patches import Patch

    updates, fills = [], []
    for a in algorithms:
        u = method_decomp(a)[1]
        if u not in updates:
            updates.append(u)
        hatch, label = _bar_fill(a)
        if (hatch, label) not in fills:
            fills.append((hatch, label))

    hue_row = [Patch(facecolor=_bar_colour_for(u), edgecolor="white",
                     label=UPDATE_LABEL.get(u, u)) for u in updates]
    order = [lab for _, lab in BAR_FILL.values()]
    fill_row = [Patch(facecolor="0.55", hatch=h, edgecolor="white", label=lab)
                for h, lab in sorted(fills, key=lambda hl: order.index(hl[1]))]
    _legend_rows(fig, [hue_row, fill_row])


def make_figure(results, sweep, algorithms, output_dir=OUTPUT_DIR, hp=None):
    import matplotlib.pyplot as plt

    if sweep.facet == "update":
        seen = {}
        for alg in algorithms:
            seen.setdefault(_axes_of(alg, hp, sweep.style)[0], []).append(alg)
        groups = [(_group_label(g), algs) for g, algs in seen.items()]
    else:
        groups = [(None, list(algorithms))]

    probe = series(results, sweep, algorithms, sweep.panels[0]["metric"],
                   sweep.panels[0]["kind"])
    categorical = isinstance(next(iter(probe.values()))[0][0], str)
    nrow, ncol = len(sweep.panels), len(groups)
    side_by_side = ncol == 1 and nrow > 1
    if side_by_side:
        nrow, ncol = 1, len(sweep.panels)

    fig, axes = plt.subplots(nrow, ncol, squeeze=False, sharex=True,
                             sharey=False if side_by_side
                                    else ("row" if ncol > 1 else False),
                             figsize=((6.5 if categorical else
                                       (3.6 if side_by_side else 4.2)) * ncol,
                                      (3.1 if side_by_side else 4.2) * nrow))
    fig.patch.set_facecolor("white")

    for r, panel in enumerate(sweep.panels):
        for c, (title, algs) in enumerate(groups):
            ax = axes[0][r] if side_by_side else axes[r][c]
            data = series(results, sweep, algs, panel["metric"], panel["kind"])
            _draw_panel(ax, data, algs, sweep, panel, categorical,
                        legend=False, hp=hp)
            if title and r == 0:
                ax.set_title(title, fontsize=14)
            if not side_by_side:
                if c > 0:
                    ax.set_ylabel("")
                if r < nrow - 1:
                    ax.set_xlabel("")
    fig.tight_layout(w_pad=2.0 if side_by_side else 1.08)
    if categorical:
        _bar_legend(fig, algorithms, hp)
    else:
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


def make_combined_figure(blobs, output_dir, name):
    import matplotlib.pyplot as plt

    sweeps = [load_sweep(dict(b["sweep"])) for b in blobs]
    algorithms, hp = [], {}
    for b in blobs:
        for a in b["algorithms"]:
            if a not in algorithms:
                algorithms.append(a)
        hp.update(b.get("hyperparams") or {})

    fig, axes = plt.subplots(1, len(blobs), squeeze=False,
                             figsize=(3.6 * len(blobs), 3.1))
    fig.patch.set_facecolor("white")

    for i, (blob, sweep) in enumerate(zip(blobs, sweeps)):
        panel = sweep.panels[0]
        algs = blob["algorithms"]
        data = series(blob["results"], sweep, algs, panel["metric"], panel["kind"])
        categorical = isinstance(next(iter(data.values()))[0][0], str)
        _draw_panel(axes[0][i], data, algs, sweep, panel, categorical,
                    legend=False, hp=hp, style_algs=algorithms)
    fig.tight_layout(w_pad=2.0)
    _split_legend(fig, algorithms, hp, faceted=False, style=sweeps[0].style)

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / name
    for ext in ("pdf", "png"):
        fig.savefig(f"{stem}.{ext}", dpi=300, bbox_inches="tight")
    print(f"Saved {stem}.pdf / {stem}.png")
    return fig


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results", nargs="+",
                    help="sweep_<key>.json, or a dir holding one. Several files "
                         "draw one panel each into a single figure (needs --name)")
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--algorithms", nargs="+", default=None,
                    help="subset/order to plot (default: what the sweep ran)")
    ap.add_argument("--style", choices=("rho", "proposal"), default=None,
                    help="what the line style encodes (default: as run)")
    ap.add_argument("--facet", choices=("update", "none"), default=None,
                    help="one column per update rule, or a single axes")
    ap.add_argument("--name", default=None, help="output figure stem")
    args = ap.parse_args()

    paths = []
    for raw in args.results:
        path = Path(raw)
        if path.is_dir():
            found = sorted(path.glob("sweep_*.json"))
            if len(found) != 1:
                raise SystemExit(f"expected one sweep_*.json in {path}, found {found}")
            path = found[0]
        paths.append(path)

    from geom_runner import code_hash
    for path in paths:
        blob = json.load(open(path))
        if blob.get("code") and blob["code"] != code_hash():
            print(f"  ! {path}: produced by different code "
                  f"({blob['code']} vs {code_hash()} now)")

    if len(paths) > 1:
        if not args.name:
            raise SystemExit("--name is required when combining several sweeps")
        blobs = [json.load(open(q)) for q in paths]
        for blob in blobs:
            print_table(blob["results"], load_sweep(dict(blob["sweep"])),
                        blob["algorithms"], blob.get("hyperparams"))
        make_combined_figure(blobs, Path(args.output_dir) if args.output_dir
                             else OUTPUT_DIR, args.name)
        return

    path = paths[0]
    blob = json.load(open(path))
    blk = dict(blob["sweep"])
    if args.style:  blk["style"] = args.style
    if args.facet:  blk["facet"] = None if args.facet == "none" else args.facet
    if args.name:   blk["name"] = args.name
    sweep = load_sweep(blk)
    algorithms = args.algorithms or blob["algorithms"]
    results = blob["results"]

    hp = blob.get("hyperparams")
    print_table(results, sweep, algorithms, hp)
    make_figure(results, sweep, algorithms,
                Path(args.output_dir) if args.output_dir else OUTPUT_DIR, hp)


if __name__ == "__main__":
    main()
