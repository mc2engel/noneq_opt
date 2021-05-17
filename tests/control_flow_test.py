"""Tests for `noneq_opt.control_flow`."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from noneq_opt import control_flow

class TestCheckpointScan:

  @pytest.mark.parametrize(
    ['f', 'init', 'xs'],
    [
      (jnp.add, jnp.zeros([3, 4]), jnp.arange(30)),
      (jnp.multiply, jnp.ones([5, 6, 7]), jnp.linspace(.5, 1.5, 30))
    ]
  )
  @pytest.mark.parametrize(
    ['checkpoint_every'],
    [(1,), (3,), (5,)]
  )
  def test_equivalent_to_scan(self, f, init, xs, checkpoint_every):
    def step(carry, x):
      y = f(carry, x)
      return y, y
    _, expected = jax.lax.scan(step, init, xs)
    _, actual = control_flow.checkpoint_scan(step, init, xs, checkpoint_every)
    np.testing.assert_allclose(expected, actual)

