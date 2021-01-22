"""Tests for `noneq_opt.parameterization`."""

import pytest

import jax
import jax.numpy as jnp
import numpy as np

import noneq_opt.parameterization as p10n

p10ns = [
    p10n.Constant(jnp.pi),
    p10n.PiecewiseLinear(jnp.linspace(-4, 6, 200)),
    p10n.Chebyshev(jnp.zeros(32)),
]


class TestParameterization:

    @pytest.mark.parametrize("parameterization", p10ns)
    def testFinite(self, parameterization):
        """Test that p10ns produce real results on [0, 1]"""
        xs = jnp.linspace(0, 1, 100)
        ys = parameterization(xs)
        print(xs, ys)
        assert np.isfinite(ys).all()

    @pytest.mark.parametrize("parameterization", p10ns)
    def testFlattenUnflatten(self, parameterization):
        """Test that p10ns can be flattened and unflattened by JAX."""
        flattened, structure = jax.tree_flatten(parameterization)
        unflattened = jax.tree_unflatten(structure, flattened)
        import pdb; pdb.set_trace()
        assert parameterization == unflattened
