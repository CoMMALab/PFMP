import os
import json
from collections import OrderedDict
import numpy as np
import jax

import numpy as np
from typing import Iterable, Tuple

from .file_utils import load_trajs_npz_named
from tqdm import tqdm

def pareto_frontier(
    points: Iterable[Tuple[float, float, float]],
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
    order = np.lexsort((P[:, 0], P[:, 2]))
    P_sorted = P[order]
    print(P_sorted)
    # done

    frontier = []
    frontier_idx = []
    best_cost = np.inf
    for j, (cost, acost, rt) in enumerate(P_sorted):
        print(j, rt, cost)
        # keep if this point strictly improves the best (lowest) cost so far
        if cost <= best_cost - eps_cost:
            frontier.append((rt, cost))
            frontier_idx.append(order[j])
            best_cost = cost

    frontier = np.array(frontier)
    if return_indices:
        return frontier, np.array(frontier_idx, dtype=int)
    return frontier


# Example usage
data = [(10, 10, 0.5), (8, 8, 0.4), (6, 6, 0.7), (4, 4, 0.6), (3, 3, 0.9)]
frontier = pareto_frontier(data)
print(frontier)
# → array of Pareto-optimal (runtime, reward) points


# need best cost, average cost, runtime

def get_best_and_average_cost(filepath):
    traj_costs = load_trajs_npz_named(filepath)['traj_costs']
    total_costs = traj_costs.reshape(traj_costs.shape[0], -1).sum(axis=1)

    terminal_costs = traj_costs.reshape(traj_costs.shape[0], -1)[:, -1]



    best_cost = np.min(terminal_costs)
    average_cost = np.mean(terminal_costs)

    # best_cost = np.min(total_costs)
    # average_cost = np.mean(total_costs)

    # return best_cost, average_cost, np.argmin(total_costs)
    return best_cost, average_cost, np.argmin(terminal_costs)
    # map_traj = jax.tree.map(lambda x: x[map_idx].reshape(self.config.H*self.config.L,-1), trajs)


def get_result_from_file(filepath, return_best_traj = False):

    json_data = None
    if filepath.endswith(".json"):
        with open(filepath, 'r') as fp:
            json_data = json.load(fp)
    else:
        return None, None

    metadata = json_data["meta"]
    results = json_data["results"]
    config = json_data["results"]["config"]

    best_cost, average_cost, best_idx = get_best_and_average_cost(os.path.join(os.path.dirname(filepath), metadata["traj_costs_fname"]))

    k, deg, h, l, sigma, rho = config["K"], config["deg"], config["H"], config["L"], config["sigma"], config["rho"]
    
    runtime = results["metrics"]["runtime"]

    best_traj = None
    if return_best_traj:
        trajs = load_trajs_npz_named(os.path.join(os.path.dirname(filepath), metadata["trajs_file"]))
        best_traj = jax.tree.map(lambda x: x[best_idx].reshape(h*l,-1), trajs)

        # trajs = trajs.reshape(trajs.shape[0], -1)
        # best_traj = trajs[best_idx]


    return ((k, deg, h, l, sigma, rho),  (best_cost, average_cost, runtime, best_traj))

    




if __name__ == "__main__":
    name = "inverted_double_pendulum"
    results_dir = "experiments/drone_2d_nav_no_obs_grid_search_pfmp"
    results_map = {}

    for filename in tqdm(os.listdir(results_dir)[::1]):
            results = (get_result_from_file(os.path.join(results_dir, filename)))
            if results[0] is not None:
                results_map[results[0]] = tuple(results[1][:-1])

    # sort by cost into an OrderedDict
    sorted_results = OrderedDict(
        sorted(results_map.items(), key=lambda item: item[1])
    )

    import matplotlib.pyplot as plt

    ks = [meta[0] for meta in sorted_results]
    distinct_ks = set(ks)
    label_to_int = {label: i for i, label in enumerate(sorted(distinct_ks))}
    cmap = plt.cm.get_cmap("tab10", len(distinct_ks))
    label_to_color = {label: cmap(i) for i, label in enumerate(range(0, len(distinct_ks)))}


    best_costs = [v[0] for v in sorted_results.values()]
    average_costs = [v[1] for v in sorted_results.values()]
    runtimes = [v[2] for v in sorted_results.values()]
    plt.scatter(runtimes, np.log(best_costs), c=[label_to_color[label_to_int[l]] for l in ks], s = 10)

    for label, i in label_to_int.items():
        plt.scatter([], [], color=cmap(i), label=label)
    plt.legend(title="K")

    # print(np.min(best_costs), np.max(best_costs))

    
    frontier = pareto_frontier(np.array([x for x in sorted_results.values()]))
    costs = [v[1] for v in frontier]
    runtimes = [v[0] for v in frontier]
    print(runtimes, costs)
    # plt.scatter(runtimes, np.log(costs), color="magenta", s = 10)
    plt.plot(runtimes, np.log(costs), color="magenta")
    # plt.show()
    plt.savefig("experiments/matplotlib/k.png")
    plt.close()


    ks = [meta[1] for meta in sorted_results]
    distinct_ks = set(ks)
    label_to_int = {label: i for i, label in enumerate(sorted(distinct_ks))}
    cmap = plt.cm.get_cmap("tab10", len(distinct_ks))
    label_to_color = {label: cmap(i) for i, label in enumerate(range(0, len(distinct_ks)))}


    best_costs = [v[0] for v in sorted_results.values()]
    average_costs = [v[1] for v in sorted_results.values()]
    runtimes = [v[2] for v in sorted_results.values()]
    plt.scatter(runtimes, np.log(best_costs), c=[label_to_color[label_to_int[l]] for l in ks], s = 10)

    for label, i in label_to_int.items():
        plt.scatter([], [], color=cmap(i), label=label)
    plt.legend(title="Deg")

    # print(np.min(best_costs), np.max(best_costs))

    
    frontier = pareto_frontier(np.array([x for x in sorted_results.values()]))
    costs = [v[1] for v in frontier]
    runtimes = [v[0] for v in frontier]
    print(runtimes, costs)
    # plt.scatter(runtimes, np.log(costs), color="magenta", s = 10)
    plt.plot(runtimes, np.log(costs), color="magenta")
    # plt.show()
    plt.savefig("experiments/matplotlib/deg.png")
    plt.close()

    ks = [meta[2] for meta in sorted_results]
    distinct_ks = set(ks)
    label_to_int = {label: i for i, label in enumerate(sorted(distinct_ks))}
    cmap = plt.cm.get_cmap("tab10", len(distinct_ks))
    label_to_color = {label: cmap(i) for i, label in enumerate(range(0, len(distinct_ks)))}


    best_costs = [v[0] for v in sorted_results.values()]
    average_costs = [v[1] for v in sorted_results.values()]
    runtimes = [v[2] for v in sorted_results.values()]
    plt.scatter(runtimes, np.log(best_costs), c=[label_to_color[label_to_int[l]] for l in ks], s = 10)

    for label, i in label_to_int.items():
        plt.scatter([], [], color=cmap(i), label=label)
    plt.legend(title="H")

    # print(np.min(best_costs), np.max(best_costs))

    
    frontier = pareto_frontier(np.array([x for x in sorted_results.values()]))
    costs = [v[1] for v in frontier]
    runtimes = [v[0] for v in frontier]
    print(runtimes, costs)
    # plt.scatter(runtimes, np.log(costs), color="magenta", s = 10)
    plt.plot(runtimes, np.log(costs), color="magenta")
    # plt.show()
    plt.savefig("experiments/matplotlib/H.png")
    plt.close()

    ks = [meta[3] for meta in sorted_results]
    distinct_ks = set(ks)
    label_to_int = {label: i for i, label in enumerate(sorted(distinct_ks))}
    cmap = plt.cm.get_cmap("tab10", len(distinct_ks))
    label_to_color = {label: cmap(i) for i, label in enumerate(range(0, len(distinct_ks)))}


    best_costs = [v[0] for v in sorted_results.values()]
    average_costs = [v[1] for v in sorted_results.values()]
    runtimes = [v[2] for v in sorted_results.values()]
    plt.scatter(runtimes, np.log(best_costs), c=[label_to_color[label_to_int[l]] for l in ks], s = 10)

    for label, i in label_to_int.items():
        plt.scatter([], [], color=cmap(i), label=label)
    plt.legend(title="L")

    # print(np.min(best_costs), np.max(best_costs))

    
    frontier = pareto_frontier(np.array([x for x in sorted_results.values()]))
    costs = [v[1] for v in frontier]
    runtimes = [v[0] for v in frontier]
    print(runtimes, costs)
    # plt.scatter(runtimes, np.log(costs), color="magenta", s = 10)
    plt.plot(runtimes, np.log(costs), color="magenta")
    # plt.show()
    plt.savefig("experiments/matplotlib/L.png")
    plt.close()


    ks = [meta[4] for meta in sorted_results]
    distinct_ks = set(ks)
    label_to_int = {label: i for i, label in enumerate(sorted(distinct_ks))}
    cmap = plt.cm.get_cmap("tab10", len(distinct_ks))
    label_to_color = {label: cmap(i) for i, label in enumerate(range(0, len(distinct_ks)))}


    best_costs = [v[0] for v in sorted_results.values()]
    average_costs = [v[1] for v in sorted_results.values()]
    runtimes = [v[2] for v in sorted_results.values()]
    plt.scatter(runtimes, np.log(best_costs), c=[label_to_color[label_to_int[l]] for l in ks], s = 10)

    for label, i in label_to_int.items():
        plt.scatter([], [], color=cmap(i), label=label)
    plt.legend(title="sigma")

    # print(np.min(best_costs), np.max(best_costs))

    
    frontier = pareto_frontier(np.array([x for x in sorted_results.values()]))
    costs = [v[1] for v in frontier]
    runtimes = [v[0] for v in frontier]
    print(runtimes, costs)
    # plt.scatter(runtimes, np.log(costs), color="magenta", s = 10)
    plt.plot(runtimes, np.log(costs), color="magenta")
    # plt.show()
    plt.savefig("experiments/matplotlib/sigma.png")
    plt.close()

    ks = [meta[5] for meta in sorted_results]
    distinct_ks = set(ks)
    label_to_int = {label: i for i, label in enumerate(sorted(distinct_ks))}
    cmap = plt.cm.get_cmap("tab10", len(distinct_ks))
    label_to_color = {label: cmap(i) for i, label in enumerate(range(0, len(distinct_ks)))}


    best_costs = [v[0] for v in sorted_results.values()]
    average_costs = [v[1] for v in sorted_results.values()]
    runtimes = [v[2] for v in sorted_results.values()]
    plt.scatter(runtimes, np.log(best_costs), c=[label_to_color[label_to_int[l]] for l in ks], s = 10)

    for label, i in label_to_int.items():
        plt.scatter([], [], color=cmap(i), label=label)
    plt.legend(title="rho")

    # print(np.min(best_costs), np.max(best_costs))

    
    frontier = pareto_frontier(np.array([x for x in sorted_results.values()]))
    costs = [v[1] for v in frontier]
    runtimes = [v[0] for v in frontier]
    print(runtimes, costs)
    # plt.scatter(runtimes, np.log(costs), color="magenta", s = 10)
    plt.plot(runtimes, np.log(costs), color="magenta")
    # plt.show()                    
    plt.savefig("experiments/matplotlib/rho.png")
    plt.close()
