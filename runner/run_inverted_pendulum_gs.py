import os
from runner.expt_runner import Problem
from tests.mjx_mod.models.inverted_double_pendulum import linear_proposal_fn, cubic_proposal_fn, quintic_proposal_fn, cost_fn, dynamics_fn


from runner.grid_search_runner import GridSearch

if __name__ == "__main__":

    problem = Problem(
        name = "inverted_double_pendulum",
        linear_proposal_fn=linear_proposal_fn,
        cubic_proposal_fn=cubic_proposal_fn,
        quintic_proposal_fn=quintic_proposal_fn,
        cost_fn=cost_fn,
        dynamics_fn=dynamics_fn
    )

    gs = GridSearch(
        problem, results_dir = "experiments/try", resume_from_iteration = 514, resume = True
    )
    gs()