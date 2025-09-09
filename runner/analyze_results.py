import os
import json
from collections import OrderedDict
import numpy as np


import numpy as np
from typing import Iterable, Tuple

def pareto_frontier(
    points: Iterable[Tuple[float, float]],
    eps_cost: float = 0.0,
    return_indices: bool = False,
):
    """
    Pareto frontier for (runtime, cost) with both minimized.
    Returns points sorted by runtime ascending; along the frontier the cost
    will be strictly decreasing (by > eps_cost).
    """
    P = np.asarray(points, dtype=float)
    if P.size == 0:
        return (P, np.empty((0,), dtype=int)) if return_indices else P

    # Sort by runtime ↑, then cost ↑ so the first time we see a runtime
    # we also see the minimum cost at that runtime (prevents dominated ties).
    order = np.lexsort((P[:, 1], P[:, 0]))
    P_sorted = P[order]

    frontier = []
    frontier_idx = []
    best_cost = np.inf
    for j, (rt, cost) in enumerate(P_sorted):
        # keep if this point strictly improves the best (lowest) cost so far
        if cost < best_cost - eps_cost:
            frontier.append((rt, cost))
            frontier_idx.append(order[j])
            best_cost = cost

    frontier = np.array(frontier)
    if return_indices:
        return frontier, np.array(frontier_idx, dtype=int)
    return frontier


# Example usage
data = [(10, 0.5), (8, 0.4), (6, 0.7), (4, 0.6), (3, 0.2)]
frontier = pareto_frontier(data)
print(frontier)
# → array of Pareto-optimal (runtime, reward) points


if __name__ == "__main__":
    name = "inverted_double_pendulum"
    results_dir = "experiments/spline_grid"
    results_map = {}

    for filename in os.listdir(results_dir):
        if filename.endswith(".json"):
            with open(os.path.join(results_dir, filename), 'r') as fp:
                results = json.load(fp)["results"]

            param_str = (
                filename.removeprefix(name + '_')
                        .removesuffix(".json")
            )
            params = [v[1:].replace('p', '.') for v in param_str.split('_')]
            (k, deg, h, l) = (int(v) for v in params[:-1])
            sigma = float(params[-1])

            cost = results["average_cost"]
            runtime = results["runtime"]
            results_map[(k, deg, h, l, sigma)] = (cost, runtime)

    # sort by cost into an OrderedDict
    sorted_results = OrderedDict(
        sorted(results_map.items(), key=lambda item: item[1])
    )

    import matplotlib.pyplot as plt

    costs = [v[0] for v in sorted_results.values()]
    runtimes = [v[1] for v in sorted_results.values()]
    plt.scatter(runtimes, costs)

    
    frontier = pareto_frontier(np.array([x for x in sorted_results.values()]))
    costs = [v[0] for v in frontier]
    runtimes = [v[1] for v in frontier]
    plt.scatter(runtimes, costs, color="magenta")
    plt.plot(runtimes, costs, color="magenta")
    plt.show()