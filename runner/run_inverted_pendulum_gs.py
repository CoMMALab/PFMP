import os
# from runner.expt_runner import Problem

from tests.models.inverted_double_pendulum import InvertedDoublePendulum
from tests.models.panda_t_pushing import TPusher
from tests.models.drone_2d_nav import Drone2DNav


from runner.grid_search_runner import GridSearch

if __name__ == "__main__":

    # problem = Problem(
    #     name = "inverted_double_pendulum",
    #     linear_proposal_fn=linear_proposal_fn,
    #     cubic_proposal_fn=cubic_proposal_fn,
    #     quintic_proposal_fn=quintic_proposal_fn,
    #     cost_fn=cost_fn,
    #     dynamics_fn=dynamics_fn
    # )
    problem = Drone2DNav()

    gs = GridSearch(
        problem, results_dir = "experiments/drone_2d_nav_grid_search_pfmp", resume_from_iteration = 0, resume = True
    )
    gs()