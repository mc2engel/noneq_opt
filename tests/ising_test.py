"""Tests for `noneq_opt.parameterization`."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from noneq_opt import ising

class TestLibraryFunctions:

  @pytest.mark.parametrize(
    ['input', 'expected'],
    [(jnp.array([0, 1, 2]),
      jnp.array([3, 2, 1])),
     (jnp.array([[0, 1, 2],
                 [3, 4, 5],
                 [6, 7, 8]]),
      jnp.array([[12, 13, 14],
                 [15, 16, 17],
                 [18, 19, 20]]))]
  )
  def test_sum_neighbors(self, input, expected):
    np.testing.assert_array_equal(ising.sum_neighbors(input), expected)

  @pytest.mark.parametrize(
    ['spins', 'log_temp', 'field', 'expected'],
    [
      (jnp.array([-1, 1, 1, 1, -1]), 0., 1., jnp.array([2., -2., -6., -2., 2.])),
      (jnp.array([-1, 1, 1, 1, -1]), 0., -1., jnp.array([-2., 2., -2., 2., -2.])),
      (jnp.array([-1, 1, 1, 1, -1]), jnp.log(2.), 1., jnp.array([1., -1., -3., -1., 1.])),
      (jnp.array([-1, 1, 1]), 100., -1., jnp.array([0., 0., 0.])),
    ]
  )
  def test_flip_logits(self, spins, log_temp, field, expected):
    params = ising.IsingParameters(log_temp, field)
    np.testing.assert_allclose(expected, ising.flip_logits(spins, params))

  @pytest.mark.parametrize(
    ['shape', 'expected_even', 'expected_odd'],
    [
      ((2, 2),
       jnp.array([[0, 1],
                  [1, 0]]),
       jnp.array([[1, 0],
                  [0, 1]])),
      ((4, 2),
       jnp.array([[0, 1],
                  [1, 0],
                  [0, 1],
                  [1, 0]]),
       jnp.array([[1, 0],
                  [0, 1],
                  [1, 0],
                  [0, 1]])),
    ]
  )
  def test_even_odd_mask(self, shape, expected_even, expected_odd):
    actual_even, actual_odd = ising.even_odd_masks(shape)
    np.testing.assert_array_equal(expected_even, actual_even)
    np.testing.assert_array_equal(expected_odd, actual_odd)


class TestSimulation:
  @pytest.mark.parametrize(
    ['shape', 'field', 'temperature', 'seed'],
    [([10000], 0., 1., 0),
     ([256, 256], .3, .5, 0),
     ([64, 64, 64], -.7, 1.5, 0)]
  )
  def test_detailed_balance(self, shape, field, temperature, seed):
    init_seed, simulation_seed = jax.random.split(jax.random.PRNGKey(seed))
    params = ising.IsingParameters(jnp.log(temperature), field)

    init_spins = ising.random_spins(shape, .5, init_seed)
    init_state = ising.IsingState(init_spins, params)
    init_log_prob = - ising.energy(init_state) / temperature

    new_state, summary = ising.update(init_state, params, simulation_seed)
    new_log_prob = - ising.energy(new_state) / temperature

    np.testing.assert_allclose(init_log_prob + summary.forward_log_prob,
                               new_log_prob + summary.reverse_log_prob,
                               rtol=1e-4)


  @pytest.mark.parametrize(
    ['shape', 'field', 'temperature', 'expected_proportion', 'seed'],
    [([10000], 0., 1., .5, 0),
     ([256, 256], 3., .2, 1., 0),
     ([64, 64, 64], -4., .5, 0., 0)]
  )
  def test_statistics(self, shape, field, temperature, expected_proportion, seed):
    """Sanity check Ising simulations by checking the proprotion of spins for simple examples."""
    init_seed, simulation_seed = jax.random.split(jax.random.PRNGKey(seed), 2)
    init_spins = ising.random_spins(shape, .5, init_seed)
    # Run for 1000 steps.
    schedule = ising.map_stack([ising.IsingParameters(jnp.log(temperature), field)] * 1000)
    state, _ = ising.simulate_ising(schedule, init_spins, simulation_seed)
    actual_proportion = ((state.spins + 1) / 2).mean()
    print(f'PROPORTIONS: {expected_proportion}, {actual_proportion}')
    np.testing.assert_allclose(actual_proportion, expected_proportion, atol=.05)

