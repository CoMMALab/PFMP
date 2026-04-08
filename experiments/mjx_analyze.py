import os
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
from pfmp.run_batch import Trajectory
from fire import Fire

# METHODS = ["ps", "mppi", "cem", "sis", "sir"]
# LABELS = [
#     "Hydrax (PS)",
#     "Hydrax (MPPI)",
#     "Hydrax (CEM)",
#     "PFMP (SIS)",
#     "PFMP (SIR)"
# ]
METHODS = ["sir"]
LABELS = [
    "PFMP (SIR)"
]

def main(results):
    trajs = [Trajectory.load(os.path.join(results, m + ".npz")) for m in METHODS]
    cost_data = [traj.cost.reshape(-1) for traj in trajs]
    
    # Table
    rows = []
    for label, data in zip(LABELS, cost_data):
        rows.append([label, f"{np.mean(data):.3f}", f"{np.std(data):.3f}"])
    
    print(tabulate(rows, headers=["Method", "Mean", "Std"], tablefmt="latex_booktabs"))
    
    # Plot
    for i, y in enumerate(cost_data):
        x = np.full(len(y), float(i)) + 0.05 * np.random.randn(len(y))
        plt.scatter(x, y, s=8, alpha=0.6)
    
    plt.xticks(range(len(LABELS)), LABELS)
    plt.title(os.path.basename(results))
    plt.xlabel("Method")
    plt.ylabel("Cost")
    plt.show()

if __name__ == "__main__":
    Fire(main)