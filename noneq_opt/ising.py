"""Code for running and optimizing Ising simulations."""

from typing import Iterable, NamedTuple, Tuple, Callable

import jax
import jax.numpy as jnp

from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions

from . import parameterization

class IsingParameters(NamedTuple):
  log_temp: jnp.array
  field: jnp.array


class IsingState(NamedTuple):
  spins: jnp.array
  params: IsingParameters


class IsingSummary(NamedTuple):
  work: jnp.array
  forward_log_prob: jnp.array
  reverse_log_prob: jnp.array


def map_slice(x, idx):
  return jax.tree_map(lambda y: y[idx], x)


def map_stack(xs, axis=0):
  return jax.tree_multimap(lambda *args: jnp.stack(args, axis), *xs)


def random_spins(shape, p, seed):
  return 2 * tfd.Bernoulli(probs=p).sample(shape, seed=seed) - 1


def sum_neighbors(spins: jnp.array) -> jnp.array:
  """Sum over the neighbors of each point in a periodic grid."""
  # TODO: replace with efficient extract_patches method.
  ndim = spins.ndim
  rolled = []
  for axis in range(ndim):
    for shift in [-1, 1]:
      rolled.append(jnp.roll(spins, shift, axis))
  return sum(rolled)


def energy(state: IsingState):
  return (- state.spins * (sum_neighbors(state.spins) / 2 + state.params.field)).sum()


def flip_logits(spins: jnp.array,
                params: IsingParameters):
  """Compute the log odds of a flip at each site based on Glauber dynamics."""
  return - 2 * spins * (sum_neighbors(spins) + params.field) * jnp.exp(-params.log_temp)


def even_odd_masks(shape: Iterable[int]) -> Tuple[jnp.array, jnp.array]:
  """Return 'even' and 'odd' masks for (even) `shape`."""
  for s in shape:
    if s % 2:
      raise ValueError(f'All entries in `shape` must be even; got {shape}.')
  grid = jnp.meshgrid(*[jnp.arange(s) for s in shape], indexing='ij')
  mask_even = sum(grid) % 2
  return mask_even, 1 - mask_even


def masked_update(state: IsingState,
                  new_params: IsingParameters,
                  mask: jnp.array,
                  seed: jnp.array) -> Tuple[IsingState, jnp.array, jnp.array]:
  """Compute random update and energy change for sites indicated by `mask`."""
  logits = flip_logits(state.spins, new_params)
  flip_distribution = tfd.Bernoulli(logits=logits)
  flips = flip_distribution.sample(seed=seed) * mask
  forward_log_prob = (flip_distribution.log_prob(flips) * mask).sum()
  new_state = IsingState(state.spins * (1 - 2 * flips), new_params)

  # I believe `reverse_logits` should be computed using `new_params`.
  # reverse_logits = get_flip_logits(new_state, state.params)
  reverse_logits = flip_logits(new_state.spins, new_params)
  reverse_distribution = tfd.Bernoulli(logits=reverse_logits)
  reverse_log_prob = (reverse_distribution.log_prob(flips) * mask).sum()

  return new_state, forward_log_prob, reverse_log_prob


def update(state: IsingState,
           new_params: IsingParameters,
           seed: jnp.array) -> Tuple[IsingState, IsingSummary]:
  seed_even, seed_odd = jax.random.split(seed, 2)
  mask_even, mask_odd = even_odd_masks(state.spins.shape)
  # TODO: can we combine calculations of log_prob and work for efficiency?
  work = energy(IsingState(state.spins, new_params) - energy(state))
  state, even_fwd_log_prob, even_rev_log_prob = masked_update(state, new_params, mask_even, seed_even)
  state, odd_fwd_log_prob, odd_rev_log_prob = masked_update(state, new_params, mask_odd, seed_odd)
  summary = IsingSummary(work=work,
                         forward_log_prob=even_fwd_log_prob + odd_fwd_log_prob,
                         reverse_log_prob=even_rev_log_prob + odd_rev_log_prob)
  return state, summary


@jax.jit
def simulate_ising(schedule: IsingParameters,
                   initial_spins: jnp.array,
                   seed: jnp.array
  ) -> Callable[[IsingState, Tuple[IsingState, jnp.array]], Tuple[IsingState, IsingSummary]]:
  initial_state = IsingState(initial_spins, map_slice(schedule, 0))
  schedule_tail = map_slice(schedule, slice(1, None))
  seeds = jax.random.split(seed, schedule.field.shape[0] - 1)
  schedule_seeds = (schedule_tail, seeds)
  def _step(state, params_seed):
    params, seed = params_seed
    new_state, summary  = update(state, params, seed)
    return new_state, summary
  return jax.lax.scan(_step, initial_state, schedule_seeds)
