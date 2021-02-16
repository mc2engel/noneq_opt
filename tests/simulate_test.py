import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax_md import space

from noneq_opt import simulate


class TestPotentialFunctions:

  @pytest.mark.parametrize(
    ['energy_fn', 'dt', 'T_schedule', 'gamma', 'dimension', 'num_particles', 'num_steps'],
    [
      (lambda x, t: - (t + 1.) * jnp.linalg.norm(x, axis=-1).sum(), 1e-3, 1., 1e-1, 3, 111, 35),
    ]
  )
  def test_shapes(self, energy_fn, dt, T_schedule, gamma, dimension, num_particles, num_steps):
    """Exercise Brownian simulator and check shapes."""
    key = jax.random.PRNGKey(9)
    _, shift_fn = space.free()
    init_fn, apply_fn = simulate.brownian(energy_fn, shift_fn, dt, T_schedule, gamma)
    #apply_fn = jax.jit(apply_fn)
    state = init_fn(key, jnp.ones([num_particles, dimension]))
    position_shape = state.position.shape
    log_prob_shape = (num_particles,)
    t = 0.
    for _ in range(num_steps):
      state = apply_fn(state, t)
      t += dt
      np.testing.assert_equal(position_shape, state.position.shape)
      np.testing.assert_equal(log_prob_shape, state.log_prob.shape)
    pass


  @pytest.mark.parametrize(
    ['energy_fn', 'dt', 'T_schedule', 'gamma', 'dimension'],
    [
      (lambda x, t: - (t + 1.) * jnp.linalg.norm(x, axis=-1).sum(), 1e-3, 1., 1e-1, 3),
    ]
  )
  def test_gradients(self, energy_fn, dt, T_schedule, gamma, dimension):
    """Test that gradients are non-zero. This does not check for correctness of the gradients."""
    key = jax.random.PRNGKey(9)
    _, shift_fn = space.free()
    init_fn, apply_fn = simulate.brownian(energy_fn, shift_fn, dt, T_schedule, gamma)
    @functools.partial(jax.grad, has_aux=True)
    @jax.jit
    def log_prob_grad(position):
      state = init_fn(key, position)
      new_state = apply_fn(state, 0.)
      return new_state.log_prob.sum(), new_state

    grad, state = log_prob_grad(jnp.ones(dimension))
    assert (grad != 0.).all()