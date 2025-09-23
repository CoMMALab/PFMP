import os
from runner.arg_utils import build_parser_and_parse_args
from runner.expt_runner import Config, ExperimentRunner
from typing import NamedTuple

from tests.models.inverted_double_pendulum import InvertedDoublePendulum
from tests.models.panda_t_pushing import TPusher
from tests.models.drone_2d_nav import Drone2DNav

from .analyze_results import get_result_from_file

from .file_utils import load_trajs_npz_named

def main():
    results_dir = "experiments/drone_2d_nav_grid_search_pfmp"


    for filename in os.listdir(results_dir):

    # filename = "panda_t_push_k1000_d1_h20_l80_s0p125.json"
        try:
            config, results = get_result_from_file(
                os.path.join(results_dir, filename), return_best_traj = True
            )
            if config is None:
                continue
            best_traj = results[-1]



            config = Config(
                        K = config[0],
                        H = config[2],
                        L = config[3],
                        sigma = config[4],
                        deg = config[1],
                        rho = 0.8
                        )
            print(config, results[0], results[1], results[2])
            problem = Drone2DNav()
            expt_runner = ExperimentRunner(
                config=config, problem=problem, results_dir = ""
            )

            expt_runner.reset_experiment()
            expt_runner.visualize(
                best_traj
            )
        except:
            continue


if __name__ == '__main__':
    main()