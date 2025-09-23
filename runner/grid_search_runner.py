import os
from runner.arg_utils import build_parser_and_parse_args
from runner.expt_runner import Config, ExperimentRunner
from runner.file_utils import _to_jsonable
from typing import NamedTuple, List
import itertools
from functools import reduce
from tqdm import tqdm
import time

class ConfigRange(NamedTuple):
    deg: List
    K: List
    HL: List
    sigma: List
    rho: List

    def total_size(self) -> int:
        return reduce(lambda x, y: x * len(y), list(self), 1)



INVERTED_DOUBLE_PENDULUM_CONFIG_RANGE = ConfigRange(
    K = [2**n for n in range(11,17)] + [100000],
    deg = [1, 3, 5],
    HL = [[2**n, 2**(10-n)] for n in range(5,11)],
    sigma = [2**n for n in range(-4,2)],
    rho = [0.6, 0.8]
)

T_PUSH_CONFIG_RANGE = ConfigRange(
    K = [100, 1000, 5000, 10000],
    deg = [1, 3, 5],
    HL = [[10, 160], [20, 80], [40, 40], [80, 20]],
    sigma = [2**n for n in range(-6,-2)],
    rho = [0.6, 0.8]
)

DRONE_FLAPPY_CONFIG_RANGE = ConfigRange(
    K = [100, 500, 1000, 5000, 10000, 50000],
    deg = [1, 3, 5],
    HL = [[10, 160], [20, 80], [40, 40], [80, 20], [160, 10]],
    sigma = [2**n for n in range(-6,6)],
    rho = [0.6, 0.8]
)


class GridSearch:
    def __init__(self, problem, results_dir: str, resume: bool = True, resume_from_iteration: int = 0):
        """
        Setup the different ranges
        """
        self.config_ranges = None
        if problem.name == "inverted_double_pendulum":
            self.config_ranges = INVERTED_DOUBLE_PENDULUM_CONFIG_RANGE
        elif problem.name == "panda_t_push":
            self.config_ranges = T_PUSH_CONFIG_RANGE
        elif problem.name == "drone_2d_nav":
            self.config_ranges = DRONE_FLAPPY_CONFIG_RANGE

        self.problem = problem
        self.results_dir = results_dir

        self.total_num_expts = self.config_ranges.total_size()
        self.resume = resume
        self.resume_from_iteration = resume_from_iteration if self.resume == False else 0
        self.config_generator = self.config_generator_fn()


    def config_generator_fn(self):
        for K, deg, (H, L), sigma, rho in itertools.product(self.config_ranges.K, self.config_ranges.deg, self.config_ranges.HL, self.config_ranges.sigma, self.config_ranges.rho):
            config = Config(
                deg = deg,
                K = K,
                H = H,
                L = L,
                sigma = sigma,
                rho = rho,
            )
            yield config

    def setup_experiment(self, config: Config):
        current_experiment = ExperimentRunner(
            config = config,
            problem = self.problem,
            results_dir = self.results_dir
        )
        current_experiment.reset_experiment()
        return current_experiment

    def run_single_experiment(self, current_experiment: ExperimentRunner):
        current_experiment.evaluate(save_results = True)

    
    def __call__(self):
        start_time = time.time()
        for idx, config in enumerate(tqdm(self.config_generator, total=self.total_num_expts, desc=f"Performing grid search for {self.problem.name}")):
            expt = self.setup_experiment(config)
            if self.resume:
                json_name, json_path = expt.get_file_name_from_config()

                if json_path.exists():
                    print(f"SKIP (exists): {json_name}")
                    continue

            if idx < self.resume_from_iteration:
                continue

            # print(f"[{idx}/{self.total_num_expts}] {config}")
            self.run_single_experiment(expt)

        print(f"Done {idx}/{self.total_num_expts} in {time.time() - start_time:.1f}s → {self.results_dir}")


