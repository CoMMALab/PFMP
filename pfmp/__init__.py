from typing import Literal, Optional, Tuple

from .algorithm import PFMP, \
    mjx_hydrax_cost, mjx_linear_gaussian_proposal, mjx_physics, \
    make_perturbation, sigma_at, knot_index, \
    GenericMJXTaskParams
from .update import ps_update, mppi_update, cem_update, sample_update, dial_step, dial_update, dial_sigma, ess
from .interpolate import hermite_spline
from .interactive import run_interactive
from .run_batch import run_batch
from .obstacle_grid import (
    sdf_aabb, hinge_loss, plot_obstacles,
    generate_obstacle_grid, generate_random_obstacles,
    generate_map, generate_map_batch,
    setup_2d, animate_2d,
)


ProposalType = Literal["gaussian", "informed", "learned"]
UpdateType = Literal["argmin", "mean", "topk", "diffuse", "sample"]

# A method name is a point on the three axes.
#
#   proposal  gaussian = perturb theta.mean; informed = the task's own q
#   update    argmin (PS) | mean (MPPI) | topk (CEM) | sample (posterior) |
#             diffuse (DIAL, which owns its annealing loop)
#   rho       resampling threshold; 0 = never. None = the task's ess_thresh,
#             which differs per task (MBM 0.552, push-T 0.25, drive 0.5) and may
#             be overridden by tuned params, so it is resolved at build time
#             rather than baked in here.
METHOD_ALIASES: dict[str, Tuple[ProposalType, UpdateType]] = {
    "ps":          ("gaussian", "argmin"),   # Howell et. al. (2022)
    "mppi":        ("gaussian", "mean"),     # Williams et. al. (2017)
    "cem":         ("gaussian", "topk"),     # Rubinstein & Kroese (2004)
    "dial":        ("gaussian", "diffuse"),  # Xue et. al. (2024), pinned to N_anneal = 2
    "sis":         ("informed", "argmin"),   # ours
    "sis_sample":  ("informed", "sample"),   # ours
    "biased_mppi": ("informed", "mean"),     # Trevisan & Alonso-Mora (2024)
    "biased_cem":  ("informed", "topk"),     # ours (biased_mppi, extrapolated)
    "sample":      ("learned", "sample"),    # Piche et. al. (2018)
    "greedy":      ("learned", "argmin"),
}

def method_decomp(method: str) -> Tuple[ProposalType, UpdateType, Optional[float]]:
    """Decompose a method name into (proposal, update, rho).

        "sis" -> ("informed", "argmin", 0.0)    "mppi+r0.25" -> (..., 0.25)
        "cem+r" -> (..., None)                  "sir", "pfmp" -> "sis+r"

    A bare `+r` means returns `rho = None`.
    """
    base, sep, val = method.partition("+r")
    if base in ("sir", "pfmp"):            # legacy aliases for sis + resampling
        base, sep = "sis", "+r"
    if base not in METHOD_ALIASES:
        raise ValueError(f"unknown method {method!r}; "
                         f"have {sorted(METHOD_ALIASES)} (with an optional +r suffix)")
    proposal, update = METHOD_ALIASES[base]
    rho = (float(val) if val else None) if sep else 0.0
    return proposal, update, rho
