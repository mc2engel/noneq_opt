
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax_md import space

from noneq_opt import barrier_crossing
from noneq_opt import parameterization as p10n


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
    potential = potential_fn(location_fn,
                             displacement_fn=displacement_fn,
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
      energy_fn = barrier_crossing.potential(potential_location_fn)
      simulate = barrier_crossing.simulate_barrier_crossing(energy_fn, temperature, gamma, total_time, time_steps)
      summary = simulate(jax.random.PRNGKey(0), jnp.zeros(ndim))
      np.testing.assert_equal(summary.energy.shape, (time_steps,))
      np.testing.assert_equal(summary.work.shape, (time_steps,))
      np.testing.assert_equal(summary.time.shape, (time_steps,))
      np.testing.assert_equal(summary.state.position.shape, (time_steps, ndim))


    def test_estimate_gradient(self):
      # TODO: figure out a way to validate the gradient estimates. For now, we just verify that the code runs and produces
      # non-zero values.

      grad_estimator = barrier_crossing.estimate_gradient(trap_fn=barrier_crossing.potential,
                                                          molecule=barrier_crossing.bistable_molecule(jnp.ones(1)),
                                                          x0=jnp.zeros(1),
                                                          mass=1.,
                                                          temperature=1.,
                                                          gamma=1.,
                                                          total_time=1.,
                                                          time_steps=100)
      #
      # TODO: I think this needs to be parameterized as p10n.Chebyshev(jnp.ones(1, 32)) !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      #
      location_schedule = p10n.Chebyshev(jnp.ones([32, 1]))
      grad_estimator = jax.jit(grad_estimator)
      grad, summary = grad_estimator(location_schedule, jax.random.PRNGKey(0))
      flat_grad = jax.tree_leaves(grad)
      for g in flat_grad:
        assert g.all(), f'Got zero values for gradient: {grad}.'


     # @pytest.mark.parametrize(
     #   'loss_function',
     #   [ising.total_work, ising.total_entropy_production]
     # )
     # @pytest.mark.parametrize(
     #   ['schedule', 'initial_spins', 'batch_size', 'time_steps', 'seed'],
     #   [
     #     (ising.IsingSchedule(log_temp=p10n.Constant(1.), field=p10n.Chebyshev(jnp.ones(8))),
     #      -jnp.ones([10, 10]),
     #      32,
     #      11,
     #      jax.random.PRNGKey(0)),
     #   ]
     # )
     # def test_training_step(self, loss_function, schedule, initial_spins, batch_size, time_steps, seed):
     #   """Tests that the training step runs and produces an optimizer state with finite values after 20 steps."""
     #   optimizer = jopt.adam(1e-3)
     #   state = optimizer.init_fn(schedule)
     #   train_step = ising.get_train_step(optimizer, initial_spins, batch_size, time_steps, loss_function)
     #   for step in range(20):
     #     seed, split = jax.random.split(seed)
     #     state, summary = train_step(state, step, split)
     #   for array in jax.tree_leaves(state):
     #     assert np.isfinite(array).all()