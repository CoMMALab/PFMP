import os
import numpy as np
import matplotlib.pyplot as plt
from fire import Fire
from pfmp.passive import Trajectory


def main(results):
    trajs = [Trajectory.load(os.path.join(results, method + ".npz")) for method in ["ps", "mppi", "cem", "sis", "sir"]]
    cost_data = [traj.cost.reshape(-1) for traj in trajs]
    print([np.mean(data) for data in cost_data])
    print([np.std(data) for data in cost_data])
    for i in range(5):
        y = cost_data[i]
        x = np.full(len(y), float(i))
        x += 0.05 * np.random.randn(len(y))
        plt.scatter(x, y, s=8, alpha=0.6)
    plt.xticks([0,1,2,3,4], [
        "Hydrax (PS)",
        "Hydrax (MPPI)",
        "Hydrax (CEM)",
        "PFMP (SIS)",
        "PFMP (SIR)"
    ])
    plt.title(os.path.basename(results))
    plt.xlabel("Method")
    plt.ylabel("Cost")
    plt.show()

if __name__ == "__main__":
    Fire(main)