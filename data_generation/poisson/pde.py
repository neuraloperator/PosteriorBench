"""Poisson: -lap(phi) = f with f a GRF whose alpha ~ U(1.5, 3) and tau ~ U(2, 4) are drawn per sample."""
import jax
import jax.numpy as jnp

from data_generation.common import S, grf, solve_elliptic

OBSERVED = "phi"


@jax.jit
def solve(f):
    return solve_elliptic(jnp.ones((S, S)), f)


def sample(key):
    k_alpha, k_tau, k_grf = jax.random.split(key, 3)
    alpha = jax.random.uniform(k_alpha, (), minval=1.5, maxval=3.0)
    tau = jax.random.uniform(k_tau, (), minval=2.0, maxval=4.0)
    f = grf(alpha, tau, S, k_grf)
    return {"f": f, "phi": solve(f), "alpha": alpha, "tau": tau}
