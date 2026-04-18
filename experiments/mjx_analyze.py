import os
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
from pfmp.run_batch import Trajectory
from fire import Fire

METHODS = ["ps", "mppi", "cem", "sis", "sir"]
LABELS = {
    "ps":   "PS",
    "mppi": "MPPI",
    "cem":  "CEM",
    "sis":  "PFMP (SIS)",
    "sir":  "PFMP (SIR)",
}


def _load(results_dir, method):
    path = os.path.join(results_dir, method + ".npz")
    if not os.path.exists(path):
        return None
    return Trajectory.load(path)


def main(*results_dirs, plot: bool = True):
    """Print a combined cost + timing table across one or more result directories.

    Usage:
        python experiments/mjx_analyze.py results/
        python experiments/mjx_analyze.py results/CartPole results/DoubleCartPole results/Walker

    If a single directory is given and contains no .npz files directly, its
    subdirectories are treated as per-task result directories.
    """
    if not results_dirs:
        raise ValueError("Provide at least one results directory.")

    # Auto-expand single parent dir → sorted subdirs
    if len(results_dirs) == 1:
        d = results_dirs[0]
        npz_direct = any(f.endswith(".npz") for f in os.listdir(d))
        if not npz_direct:
            subdirs = sorted(
                os.path.join(d, s) for s in os.listdir(d)
                if os.path.isdir(os.path.join(d, s))
            )
            if subdirs:
                results_dirs = tuple(subdirs)

    task_names = [os.path.basename(d.rstrip("/")) for d in results_dirs]

    # ── Collect data ─────────────────────────────────────────
    # data[method] = {task: {"mean": float, "std": float, "replan_ms": float}}
    data = {m: {} for m in METHODS}
    for d, task in zip(results_dirs, task_names):
        for m in METHODS:
            traj = _load(d, m)
            if traj is None:
                continue
            costs = traj.cost.reshape(-1)
            data[m][task] = {
                "mean":      float(np.mean(costs)),
                "std":       float(np.std(costs)),
                "replan_ms": traj.replan_ms,
            }

    # ── Build combined table ───────────────────────────────────
    flat_header = ["Method"]
    for task in task_names:
        flat_header += [f"{task} cost", f"{task} ms"]

    rows = []
    for m in METHODS:
        if not data[m]:
            continue
        row = [LABELS[m]]
        for task in task_names:
            if task in data[m]:
                d = data[m][task]
                row.append(f"{d['mean']:.3f} $\\pm$ {d['std']:.3f}")
                row.append(f"{d['replan_ms']:.1f}")
            else:
                row += ["—", "—"]
        rows.append(row)

    # ── Human-readable ────────────────────────────────────────
    print(tabulate(rows, headers=flat_header, tablefmt="rounded_outline"))
    print()

    # ── LaTeX with \multicolumn task headers ──────────────────
    n = len(task_names)
    col_spec = "l" + "rr" * n
    task_header = " & ".join(
        f"\\multicolumn{{2}}{{c}}{{{t}}}" for t in task_names
    )
    sub_header = "Method & " + " & ".join(["Cost & ms"] * n) + r" \\"
    cmidrules = " ".join(
        f"\\cmidrule(lr){{{2 + 2*i}-{3 + 2*i}}}" for i in range(n)
    )

    latex_rows = []
    for row in rows:
        latex_rows.append(" & ".join(str(c) for c in row) + r" \\")

    col_spec = "l" + "cc" * n
    sub_header = "Method & " + " & ".join([r"\multicolumn{1}{c}{Cost} & \multicolumn{1}{c}{ms}"] * n) + r" \\"

    print(r"\begin{table*}[!t]")
    print(r"\centering")
    print(f"\\begin{{tabular}}{{{col_spec}}}")
    print(r"\toprule")
    print(f" & {task_header} \\\\")
    print(cmidrules)
    print(sub_header)
    print(r"\midrule")
    print("\n".join(latex_rows))
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\vspace{0.5em}")
    print(r"\caption{Cost (mean $\pm$ std) and per-replan wall time (ms) on an RTX 3070 Ti Laptop GPU.}")
    print(r"\label{tab:mujoco}")
    print(r"\end{table*}")

    # ── Plot ──────────────────────────────────────────────────
    if not plot:
        return

    n_tasks = len(task_names)
    fig, axes = plt.subplots(1, n_tasks, figsize=(4 * n_tasks, 4), sharey=False)
    if n_tasks == 1:
        axes = [axes]

    for ax, task in zip(axes, task_names):
        methods_present = [m for m in METHODS if task in data[m]]
        for i, m in enumerate(methods_present):
            d = data[m][task]
            ax.bar(i, d["mean"], yerr=d["std"], capsize=4, label=LABELS[m])
        ax.set_xticks(range(len(methods_present)))
        ax.set_xticklabels([LABELS[m] for m in methods_present], rotation=30, ha="right")
        ax.set_title(task)
        ax.set_ylabel("Cost")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    Fire(main)
