"""Darcy flow: -div(a grad u) = 1 with binary permeability a in {3, 12} from a thresholded GRF."""
import jax
import jax.numpy as jnp

from data_generation.common import S, grf, solve_elliptic, to_node

ALPHA, TAU = 2.0, 3.0
OBSERVED = "u"


@jax.jit
def solve(a):
    coef_v = jnp.clip(to_node(a), 3.0, 12.0)
    return solve_elliptic(coef_v, jnp.ones((S, S)))


def sample(key):
    a = jnp.where(grf(ALPHA, TAU, S, key) >= 0, 12.0, 3.0)
    return {"a": a, "u": solve(a)}
