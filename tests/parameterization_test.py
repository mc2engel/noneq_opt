"""Tests for `noneq_opt.parameterization`."""

import pytest

import jax
import jax.numpy as jnp
import numpy as np

import noneq_opt.parameterization as p10n


class TestParameterization:

    @pytest.mark.parametrize(
    "parameterization",
    [
    p10n.Constant(jnp.pi),
    ]
    )
    def testFinite(self, parameterization):
        xs = jnp.linspace(0, 1, 100)
        ys = parameterization(xs)
        assert np.isfinite(ys).all()
