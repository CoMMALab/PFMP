"""iCEM: CEM with temporally-correlated (colored-noise) sampling.

iCEM (Pinneri et al. 2021) is the Cross-Entropy Method with a power-law
colored-noise proposal in place of white Gaussian noise, which produces smooth,
temporally-correlated control sequences. In the SMC decomposition this is the
`q_colored` + `U_elite` row: only the proposal changes from CEM.

We subclass hydrax's CEM and override `sample_knots` so that the per-knot noise
is correlated along the horizon via the Timmer & Koenig (1995) construction.
"""

from typing import Literal, Tuple

import jax
import jax.numpy as jnp

from hydrax.task_base import Task
from hydrax.risk import RiskStrategy
from hydrax.algs.cem import CEM, CEMParams


def colored_noise(key, shape, beta: float, axis: int = 1):
    """Zero-mean unit-std noise with power spectrum 1/f^beta along `axis`.

    beta=0 is white noise; beta in [2, 3] gives smooth, correlated sequences.
    Built in the frequency domain (Timmer & Koenig): scale the Fourier
    amplitudes by 1/f^(beta/2), then inverse-FFT.
    """
    H = shape[axis]
    f = jnp.fft.rfftfreq(H)
    scale = jnp.where(f > 0, f, jnp.inf) ** (-beta / 2.0)
    scale = scale.at[0].set(0.0)  # zero-mean (drop DC), avoids 1/0
    white = jax.random.normal(key, shape)
    Wf = jnp.fft.rfft(white, axis=axis)
    bshape = [1] * len(shape)
    bshape[axis] = scale.shape[0]
    colored = jnp.fft.irfft(Wf * scale.reshape(bshape), n=H, axis=axis)
    # Normalize so beta only changes correlation, not magnitude.
    return colored / (jnp.std(colored, axis=axis, keepdims=True) + 1e-8)


class iCEM(CEM):
    """CEM with colored-noise sampling (everything else identical to CEM)."""

    def __init__(
        self,
        task: Task,
        num_samples: int,
        num_elites: int,
        sigma_start: float,
        sigma_min: float,
        beta: float = 2.5,
        num_randomizations: int = 1,
        explore_fraction: float = 0.0,
        risk_strategy: RiskStrategy = None,
        seed: int = 0,
        plan_horizon: float = 1.0,
        spline_type: Literal["zero", "linear", "cubic"] = "zero",
        num_knots: int = 4,
        iterations: int = 1,
    ) -> None:
        super().__init__(
            task,
            num_samples=num_samples,
            num_elites=num_elites,
            sigma_start=sigma_start,
            sigma_min=sigma_min,
            num_randomizations=num_randomizations,
            explore_fraction=explore_fraction,
            risk_strategy=risk_strategy,
            seed=seed,
            plan_horizon=plan_horizon,
            spline_type=spline_type,
            num_knots=num_knots,
            iterations=iterations,
        )
        self.beta = beta

    def sample_knots(self, params: CEMParams) -> Tuple[jax.Array, CEMParams]:
        """Sample control sequences with colored noise along the horizon."""
        rng, sample_rng, explore_rng = jax.random.split(params.rng, 3)

        main_shape = (
            self.num_samples - self.num_explore,
            self.num_knots,
            self.task.model.nu,
        )
        explore_shape = (
            self.num_explore,
            self.num_knots,
            self.task.model.nu,
        )

        # Colored noise is correlated along axis=1 (the knot / horizon axis).
        main_controls = (
            params.mean + params.cov * colored_noise(sample_rng, main_shape, self.beta)
            if main_shape[0] > 0
            else jnp.empty(main_shape)
        )
        explore_controls = (
            params.mean + self.sigma_start * colored_noise(explore_rng, explore_shape, self.beta)
            if explore_shape[0] > 0
            else jnp.empty(explore_shape)
        )

        controls = jnp.concatenate([main_controls, explore_controls])
        return controls, params.replace(rng=rng)
