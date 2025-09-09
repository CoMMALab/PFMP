import os
from runner.arg_utils import build_parser_and_parse_args
from runner.expt_runner import Config, Problem, ExperimentRunner
from runner.file_utils import _to_jsonable
from typing import NamedTuple
from tests.mjx_mod.models.inverted_double_pendulum import linear_proposal_fn, cubic_proposal_fn, quintic_proposal_fn, cost_fn, dynamics_fn

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

    problem = Problem(
        name = "inverted_double_pendulum",
        linear_proposal_fn=linear_proposal_fn,
        cubic_proposal_fn=cubic_proposal_fn,
        quintic_proposal_fn=quintic_proposal_fn,
        cost_fn=cost_fn,
        dynamics_fn=dynamics_fn
    )

    expt_runner = ExperimentRunner(
        config=config, problem=problem, results_dir = "experiments/try"
    )

    expt_runner.reset_experiment()
    expt_runner.evaluate(save_results = True)


if __name__ == '__main__':
    main()