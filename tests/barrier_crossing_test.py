
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax_md import space

from noneq_opt import barrier_crossing


class TestPotentialFunctions:

  @pytest.mark.parametrize(
    'displacement_fn',
    [
      space.free()[0],
      space.periodic(3.)[0]
    ]
  )
  @pytest.mark.parametrize(
    'location_fn',
    [
      lambda t: jnp.sin(t) * jnp.ones(1),
      lambda t: jnp.exp(t) * jnp.ones(3),
    ]
  )
  @pytest.mark.parametrize(
    ['potential_fn', 'kwargs'],
    [
      (barrier_crossing.potential, dict(k=3.)),
      (barrier_crossing.bistable_molecule, dict(k_l=2., k_r=3., delta_e=1., beta=1.)),
    ]
  )
  def test_shapes(self, displacement_fn, location_fn, potential_fn, kwargs):
    key = jax.random.PRNGKey(0)
    dim = location_fn(0.).shape[0]
    potential = potential_fn(displacement_fn,
                             location_fn,
                             **kwargs)
    for t in jnp.linspace(-3, 3, 10):
      key, split = jax.random.split(key)
      x = jax.random.normal(shape=[dim], key=key)
      energy = potential(x, t)
      assert np.isfinite(energy)
      np.testing.assert_equal(energy.shape, ())
