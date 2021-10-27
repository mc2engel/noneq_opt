"""Tests for `noneq_opt.parameterization`."""
import functools

import jax
import jax.numpy as jnp
import jax.experimental.optimizers as jopt
import numpy as np
import pytest

from noneq_opt import ising
from noneq_opt import parameterization as p10n

class TestLibraryFunctions:

  @pytest.mark.parametrize(
    ['input', 'expected'],
    [
      (jnp.array([0, 1, 2]),
       jnp.array([3, 2, 1])),
      (jnp.array([[0, 1, 2],
                  [3, 4, 5],
                  [6, 7, 8]]),
       jnp.array([[12, 13, 14],
                  [15, 16, 17],
                  [18, 19, 20]]))
    ]
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

  @pytest.mark.parametrize('shape', [10000, [256, 256], [64, 64, 64]])
  @pytest.mark.parametrize('field', jnp.linspace(-1, 1, 5))
  @pytest.mark.parametrize('temperature', jnp.exp(jnp.linspace(-1, 1, 5)))
  @pytest.mark.parametrize('seed', [0])
  def test_detailed_balance(self, shape, field, temperature, seed):
    """Test that detailed balance is satisfied."""
    init_seed, simulation_seed = jax.random.split(jax.random.PRNGKey(seed))
    params = ising.IsingParameters(jnp.log(temperature), field)

    init_spins = ising.random_spins(shape, .5, init_seed)
    init_state = ising.IsingState(init_spins, params)
    init_log_prob = - ising.energy(init_state) / temperature

    new_state, summary = ising.update(init_state, params, simulation_seed)
    new_log_prob = - ising.energy(new_state) / temperature

    np.testing.assert_allclose(init_log_prob + summary.forward_log_prob,
                               new_log_prob + summary.reverse_log_prob,
                               rtol=5e-4)

  @pytest.mark.parametrize('shape', [10000, [256, 256], [64, 64, 64]])
  @pytest.mark.parametrize('field', jnp.linspace(-1, 1, 5))
  @pytest.mark.parametrize('temperature', jnp.exp(jnp.linspace(-1, 1, 5)))
  @pytest.mark.parametrize('seed', [0])
  def test_entropy_production(self, shape, field, temperature, seed):
    """Verify the identity `dissipation == temperature * (fwd_log_prob - reverse_log_prob)`."""
    init_seed, simulation_seed = jax.random.split(jax.random.PRNGKey(seed))
    params = ising.IsingParameters(jnp.log(temperature), field)

    init_spins = ising.random_spins(shape, .5, init_seed)
    init_state = ising.IsingState(init_spins, params)
    init_log_prob = - ising.energy(init_state) / temperature

    new_state, summary = ising.update(init_state, params, simulation_seed)

    # Test that the dissipated heat is equal to the entropy production times temperature.
    np.testing.assert_allclose(summary.dissipated_heat,
                               summary.entropy_production * temperature,
                               rtol=1e-4)

  @pytest.mark.parametrize(
    ['shape', 'field', 'temperature', 'expected_proportion', 'seed'],
    [
      ([10000], 0., 1., .5, 0),
      ([256, 256], 3., .2, 1., 0),
      ([64, 64, 64], -4., .5, 0., 0)
    ]
  )
  def test_statistics(self, shape, field, temperature, expected_proportion, seed):
    """Sanity check Ising simulations by checking the proprotion of spins for simple examples."""
    init_seed, simulation_seed = jax.random.split(jax.random.PRNGKey(seed), 2)
    init_spins = ising.random_spins(shape, .5, init_seed)
    # Run for 1000 steps.
    schedule = ising.map_stack([ising.IsingParameters(jnp.log(temperature), field)] * 1000)
    state, _ = jax.jit(ising.simulate_ising)(schedule, init_spins, simulation_seed)
    actual_proportion = ((state.spins + 1) / 2).mean()
    np.testing.assert_allclose(actual_proportion, expected_proportion, atol=.05)

  @pytest.mark.parametrize(
    'loss_function',
    [ising.total_work, ising.total_entropy_production]
  )
  @pytest.mark.parametrize(
    ['schedule', 'times', 'initial_spins', 'seed'],
    [
      (ising.IsingSchedule(log_temp=p10n.Chebyshev(jnp.ones(8)), field=p10n.Chebyshev(jnp.ones(8))),
       jnp.linspace(0, 1, 5),
       -jnp.ones([10, 10]),
       jax.random.PRNGKey(0)),
    ]
  )
  def test_gradient_estimates_equivalent(self, loss_function, schedule, times, initial_spins, seed):
    fwd_grad, _ = jax.jit(ising.estimate_gradient_fwd(loss_function))(schedule, times, initial_spins, seed)
    rev_grad, _ = jax.jit(ising.estimate_gradient_rev(loss_function))(schedule, times, initial_spins, seed)
    jax.tree_multimap(functools.partial(np.testing.assert_allclose, rtol=1e-4), fwd_grad, rev_grad)


  @pytest.mark.parametrize(
    'loss_function',
    [ising.total_work, ising.total_entropy_production]
  )
  @pytest.mark.parametrize(
    ['schedule', 'initial_spins', 'batch_size', 'time_steps', 'seed'],
    [
      (ising.IsingSchedule(log_temp=p10n.Constant(1.), field=p10n.Chebyshev(jnp.ones(8))),
       -jnp.ones([10, 10]),
       32,
       11,
       jax.random.PRNGKey(0)),
    ]
  )
  @pytest.mark.parametrize(
    'mode',
    ['fwd', 'rev']
  )
  def test_training_step(self, loss_function, schedule, initial_spins, batch_size, time_steps, seed, mode):
    """Tests that the training step runs and produces an optimizer state with finite values after 20 steps."""
    optimizer = jopt.adam(1e-3)
    state = optimizer.init_fn(schedule)
    train_step = ising.get_train_step(optimizer, initial_spins, batch_size, time_steps, loss_function, mode)
    for step in range(20):
      seed, split = jax.random.split(seed)
      state, summary = train_step(state, step, split)
    for array in jax.tree_leaves(state):
      assert np.isfinite(array).all()


