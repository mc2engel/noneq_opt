"""Code for running and optimizing Ising simulations."""
import functools
from typing import Iterable, NamedTuple, Optional, Tuple, Callable

import jax
import jax.experimental.optimizers as jopt
import jax.numpy as jnp
import distrax

from . import control_flow


class IsingParameters(NamedTuple):
  log_temp: jnp.array
  field: jnp.array


class IsingSchedule(NamedTuple):
  log_temp: Callable
  field: Callable

  def __call__(self, x):
    return IsingParameters(log_temp=self.log_temp(x),
                           field=self.field(x))


class IsingState(NamedTuple):
  spins: jnp.array
  params: IsingParameters


class IsingSummary(NamedTuple):
  work: jnp.array
  dissipated_heat: jnp.array
  forward_log_prob: jnp.array
  reverse_log_prob: jnp.array
  entropy_production: jnp.array
  magnetization: jnp.array
  energy: jnp.array


# TODO: move these to `utils` or similar.

def map_slice(x, idx):
  return jax.tree_map(lambda y: y[idx], x)


def map_stack(xs, axis=0):
  return jax.tree_multimap(lambda *args: jnp.stack(args, axis), *xs)


def random_spins(shape, p, seed):
  return 2 * distrax.Bernoulli(probs=p).sample(sample_shape=shape, seed=seed) - 1


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


def log_prob(state: IsingState):
  return - energy(state) / jnp.exp(state.params.log_temp)


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
  flip_distribution = distrax.Bernoulli(logits=logits)
  flips = flip_distribution.sample(seed=seed) * mask
  forward_log_prob = (flip_distribution.log_prob(flips) * mask).sum()
  new_state = IsingState(state.spins * (1 - 2 * flips), new_params)

  # I believe `reverse_logits` should be computed using `new_params`.
  # reverse_logits = get_flip_logits(new_state, state.params)
  reverse_logits = flip_logits(new_state.spins, new_params)
  reverse_distribution = distrax.Bernoulli(logits=reverse_logits)
  reverse_log_prob = (reverse_distribution.log_prob(flips) * mask).sum()

  return new_state, forward_log_prob, reverse_log_prob


def update(state: IsingState,
           new_params: IsingParameters,
           seed: jnp.array) -> Tuple[IsingState, IsingSummary]:
  seed_even, seed_odd = jax.random.split(seed, 2)
  mask_even, mask_odd = even_odd_masks(state.spins.shape)
  initial_energy = energy(state)
  intermediate_energy = energy(IsingState(state.spins, new_params))
  state, even_fwd_log_prob, even_rev_log_prob = masked_update(state, new_params, mask_even, seed_even)
  state, odd_fwd_log_prob, odd_rev_log_prob = masked_update(state, new_params, mask_odd, seed_odd)
  final_energy = energy(state)
  magnetization = state.spins.mean()
  forward_log_prob = even_fwd_log_prob + odd_fwd_log_prob
  reverse_log_prob = even_rev_log_prob + odd_rev_log_prob
  entropy_production = forward_log_prob - reverse_log_prob
  summary = IsingSummary(work=intermediate_energy - initial_energy,
                         dissipated_heat=intermediate_energy - final_energy,
                         forward_log_prob=forward_log_prob,
                         reverse_log_prob=reverse_log_prob,
                         entropy_production=entropy_production,
                         magnetization=magnetization,
                         energy=final_energy)
  return state, summary


def simulate_ising(parameters: IsingParameters,
                   initial_spins: jnp.array,
                   seed: jnp.array,
                   checkpoint_every: Optional[int] = None,
                   return_states: bool = False
  ) -> Callable[[IsingState, Tuple[IsingState, jnp.array]], Tuple[IsingState, IsingSummary]]:
  initial_state = IsingState(initial_spins, map_slice(parameters, 0))
  parameters_tail = map_slice(parameters, slice(1, None))
  seeds = jax.random.split(seed, parameters.field.shape[0] - 1)
  parameters_seeds = (parameters_tail, seeds)
  def _step(state, parameters_seed):
    parameters, seed = parameters_seed
    new_state, summary  = update(state, parameters, seed)
    if return_states:
      return new_state, (summary, new_state)
    return new_state, summary
  if checkpoint_every is None:
    scan = jax.lax.scan
  else:
    scan = functools.partial(control_flow.checkpoint_scan, checkpoint_every=checkpoint_every)
  return scan(_step, initial_state, parameters_seeds)


# A `LossFn` maps (initial state, final_state, trajectory summary) to a scalar loss.
LossFn = Callable[[IsingState, IsingState, IsingSummary], jnp.array]

def value_and_jacfwd(f, argnums=0):
  # TODO: make this only run one pass.
  jacobian = jax.jacfwd(f, argnums=argnums)
  def _val_and_jac(*args, **kwargs):
    val = f(*args, **kwargs)
    jac = jacobian(*args, **kwargs)
    return val, jac
  return _val_and_jac

def estimate_gradient(loss_function):

  def loss_and_log_prob(schedule: IsingSchedule,
                        times: jnp.array,
                        initial_spins: jnp.array,
                        seed: jnp.array
    ) -> Tuple[jnp.array, jnp.array, IsingSummary]:
    parameters = schedule(times)
    final_state, summary = simulate_ising(parameters, initial_spins, seed)
    trajectory_log_prob = summary.forward_log_prob.sum()
    initial_state = IsingState(initial_spins, map_slice(parameters, 0))
    loss = loss_function(initial_state, final_state, summary)
    total_forward_log_prob = summary.forward_log_prob.sum(0)
    return loss, total_forward_log_prob, summary

  def _estimate_gradient(schedule: IsingSchedule,
                         times: jnp.array,
                         initial_spins: jnp.array,
                         seed: jnp.array
    ) -> Tuple[IsingParameters, IsingSummary]:
    loss_and_prob_and_jacobians = value_and_jacfwd(loss_and_log_prob)
    ((loss, _, summary),
     (loss_jac, log_prob_jac, _)) = loss_and_prob_and_jacobians(schedule,
                                                                times,
                                                                initial_spins,
                                                                seed)
    # ∇_est = ∇log(p) x loss + ∇loss
    gradient_estimate = jax.tree_multimap(
        lambda x, y: x + y,
        jax.tree_map(lambda x: loss * x, log_prob_jac),
        loss_jac)
    return gradient_estimate, summary

  return _estimate_gradient



def total_entropy_production(initial_state: IsingState,
                             final_state: IsingState,
                             summary: IsingSummary) -> jnp.array:
  trajectory_entropy_production = summary.entropy_production.sum()

  # We have to include an extra term that accounts for the difference in initial and final energy/probability.
  # Note that these are unnormalized log probabilities, so this calculation assumes that the start state and
  # end state have the same partition function.
  endpoint_entropy_production = log_prob(initial_state) - log_prob(final_state)

  return endpoint_entropy_production + trajectory_entropy_production


def total_work(initial_state: IsingState,
               final_state: IsingState,
               summary: IsingSummary) -> jnp.array:
  return summary.work.sum()


# A `TrainStepFn` takes (optimizer state, step, seed) and returns (new optimizer state, summary).
TrainStepFn = Callable[[jopt.OptimizerState, jnp.array, jnp.array],
                       Tuple[jopt.OptimizerState, IsingSummary]]


def get_train_step(optimizer: jopt.Optimizer,
                   initial_spins: jnp.array,
                   batch_size: int,
                   time_steps: int,
                   loss_function: LossFn = total_entropy_production
  ) -> TrainStepFn:
  mapped_gradient_estimate = jax.vmap(estimate_gradient(loss_function), [None, None, None, 0])
  # TODO: consider taking `times` as an argument rather than assuming times in [0, 1].
  times = jnp.linspace(0, 1, time_steps)
  @jax.jit
  def _train_step(opt_state, step, seed):
    seeds = jax.random.split(seed, batch_size)
    schedule = optimizer.params_fn(opt_state)
    grads, summary = mapped_gradient_estimate(schedule, times, initial_spins, seeds)
    mean_grad = jax.tree_map(lambda x: jnp.mean(x, 0), grads)
    opt_state = optimizer.update_fn(step, mean_grad, opt_state)
    return opt_state, summary
  return _train_step