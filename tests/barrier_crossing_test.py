
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


  class TestBarrierCrossing:
    @pytest.mark.parametrize(
      ['temperature', 'gamma', 'total_time', 'time_steps', 'ndim', 'potential_location_fn'],
      [
        (1., 1., 10.,1000, 3, lambda t: t * jnp.ones(3)),
        (3., .1, 2 * jnp.pi, 100, 2, lambda t: jnp.sin(t) * jnp.ones(2)),
      ]
    )
    def test_simulate_barrier_crossing_shape_test(
        self, temperature, gamma, total_time, time_steps, ndim, potential_location_fn):
      displacement_fn, shift_fn = space.free()

      energy_fn = barrier_crossing.potential(displacement_fn, potential_location_fn)
      simulate = barrier_crossing.simulate_barrier_crossing(
        energy_fn, shift_fn, temperature, gamma, total_time, time_steps)
      summary = simulate(jax.random.PRNGKey(0), jnp.zeros(ndim))
      np.testing.assert_equal(summary.energy.shape, (time_steps,))
      np.testing.assert_equal(summary.work.shape, (time_steps,))
      np.testing.assert_equal(summary.time.shape, (time_steps,))
      np.testing.assert_equal(summary.state.position.shape, (time_steps, ndim))
