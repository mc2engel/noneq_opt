"""Tests for `noneq_opt.gradients`."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from noneq_opt import gradients

def normal(size, seed=0):
  return np.random.RandomState(seed).normal(size=size)


class TestLibraryFunctions:

  @pytest.mark.parametrize(
    ['f', 'args'],
    [
      (  # Single value -> single value
          jnp.exp,
          (normal([12, 12]),)
       ),
      (  # list -> single value
          lambda x: x[0].dot(x[1]),
          ([normal([4, 4]), normal([4])],)
      ),
      (  # list -> single value, multiple inputs
          lambda x, y, z: x[0] * y + x[1] * z,
          ([normal([2, 2], 0)] * 2, normal([2, 2], 1), normal([2, 2], 2))
      ),
      (  # array inputs -> tuple output
          lambda x, y: (x * y, x / y),
          (normal([3, 3], 0), normal([3, 3], 1))
      ),
      (  # tuple -> tuple
          lambda x: (x[0] * x[1], x[0] / x[1]),
          ((normal([3, 3], 0), normal([3, 3], 1)),)
      ),

    ]
  )
  def test_sum_neighbors(self, f, args):
    expected_value = f(*args)
    expected_jacobian = jax.jacfwd(f)(*args)
    actual_value, actual_jacobian = gradients.value_and_jacfwd(f)(*args)
    jax.tree_multimap(np.testing.assert_allclose, actual_value, expected_value)
    jax.tree_multimap(np.testing.assert_allclose, actual_jacobian, expected_jacobian)


