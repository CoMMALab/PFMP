import os
from runner.arg_utils import build_parser_and_parse_args
from runner.expt_runner import Config, ExperimentRunner
from runner.file_utils import _to_jsonable
from typing import NamedTuple
# from tests.mjx_mod.models.inverted_double_pendulum import linear_proposal_fn, cubic_proposal_fn, quintic_proposal_fn, cost_fn, dynamics_fn
from tests.models.inverted_double_pendulum import InvertedDoublePendulum
from tests.models.panda_t_pushing import TPusher

def main():
    args = build_parser_and_parse_args()
    # TODO (siyer) -- use a dict using ** directly and pass it
    config = Config(
                K = args.K,
                H = args.H,
                L = args.L,
                sigma = args.sigma,
                deg = args.deg,
                rho = args.rho
                )
    # problem = InvertedDoublePendulum()
    problem = TPusher()

    # expt_runner = ExperimentRunner(
    #     config=config, problem=inv_pend_problem, results_dir = "experiments/try_class"
    # )
    expt_runner = ExperimentRunner(
        config=config, problem=problem, results_dir = "experiments/push_grid_search"
    )

    expt_runner.reset_experiment()
    expt_runner.evaluate(save_results = True, visualize=True)


if __name__ == '__main__':
    main()