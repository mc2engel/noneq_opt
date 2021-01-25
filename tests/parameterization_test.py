"""Tests for `noneq_opt.parameterization`."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import noneq_opt.parameterization as p10n

p10ns = [
  p10n.Constant(jnp.pi),
  p10n.PiecewiseLinear(jnp.linspace(-4, 6, 200)),
  p10n.Chebyshev(jnp.zeros(32)),
  p10n.ChangeDomain(p10n.PiecewiseLinear.equidistant(10), -11., 22.),
  p10n.ConstrainEndpoints(p10n.PiecewiseLinear.equidistant(10), -11., 22.),
]


class TestParameterization:

  @pytest.mark.parametrize("parameterization", p10ns)
  def test_finite(self, parameterization):
    """Test that p10ns produce real results on [0, 1]"""
    xs = jnp.linspace(0, 1, 100)
    ys = parameterization(xs)
    assert np.isfinite(ys).all()

  @pytest.mark.parametrize("parameterization", p10ns)
  def test_flatten_unflatten(self, parameterization):
    """Test that p10ns can be flattened and unflattened by JAX."""
    flattened, structure = jax.tree_flatten(parameterization)
    unflattened = jax.tree_unflatten(structure, flattened)
    assert parameterization == unflattened


class TestChangeDomain:

  @pytest.mark.parametrize("new_x0, new_x1", [(17., 333.), (0., 100.)])
  @pytest.mark.parametrize("parameterization,", p10ns)
  def test_correct_domain(self, parameterization, new_x0, new_x1):
    moved = p10n.ChangeDomain(parameterization, x0=new_x0, x1=new_x1)
    np.testing.assert_allclose((new_x0, new_x1), moved.domain)
    np.testing.assert_allclose(moved(new_x0), parameterization(parameterization.domain[0]))
    np.testing.assert_allclose(moved(new_x1), parameterization(parameterization.domain[1]))


class TestConstrainEndpoints:

  @pytest.mark.parametrize("new_y0, new_y1", [(17., 333.), (0., 100.)])
  @pytest.mark.parametrize("parameterization,", p10ns)
  def test_endpoint_values(self, parameterization, new_y0, new_y1):
    constrained = p10n.ConstrainEndpoints(parameterization, y0=new_y0, y1=new_y1)
    x0, x1 = parameterization.domain
    np.testing.assert_allclose(parameterization.domain, constrained.domain)
    np.testing.assert_allclose(new_y0, constrained(x0))
    np.testing.assert_allclose(new_y1, constrained(x1))