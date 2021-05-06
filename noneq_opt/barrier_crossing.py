"""Code for running and optimizing Ising simulations."""
import functools
from typing import Callable, Union, NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp
import jax.experimental.optimizers as jopt
from jax_md import space, energy

from tensorflow_probability.substrates import jax as tfp

from noneq_opt import simulate

tfd = tfp.distributions

### Potential Functions ###
# These functions define potentials. They close over various parameters and return a function maps (position, time) to
# a scalar energy value.

# A location function maps time to a location vector.
Scalar = Union[jnp.array, float]
LocationFn = Callable[[jnp.array], jnp.array]
LocationFnOrConstant = Union[LocationFn, jnp.array]

# A potential function takes position and time to a scalar energy value.
PotentialFn = Callable[[jnp.array, jnp.array], jnp.array]


def map_slice(x, idx):
  return jax.tree_map(lambda y: y[idx], x)


def _get_location_fn(location: LocationFnOrConstant) -> LocationFn:
  if not callable(location):
    return lambda t: location
  return location


def potential(location_fn: LocationFnOrConstant,
              k: Scalar = 1.,
              displacement_fn: Optional[space.DisplacementFn] = None,
  ) -> PotentialFn:
  if displacement_fn is None:
    displacement_fn, _ = space.free()
  location_fn = _get_location_fn(location_fn)
  def _potential(position, t, **unused_kwargs):
    d = space.distance(displacement_fn(position, location_fn(t)))
    return energy.simple_spring(d, epsilon=k, length=0).sum()
  return _potential


def bistable_molecule(location_fn: LocationFnOrConstant,
                      k_l: Scalar = 1.,
                      k_r: Scalar = 1.,
                      delta_e: Scalar = 0.,
                      beta: Scalar = 1.,
                      displacement_fn: Optional[space.DisplacementFn] = None
  ) -> PotentialFn:
  if displacement_fn is None:
    displacement_fn, _ = space.free()
  location_fn = _get_location_fn(location_fn)
  def _bistable_molecule(position, t, **unused_kwargs):
    location = location_fn(t)
    d_l = space.distance(displacement_fn(-position, location))
    d_r = space.distance(displacement_fn(position, location))
    log_l = - beta * (k_l * jnp.square(d_l) / 2)
    log_r = - beta * (k_r * jnp.square(d_r) / 2 + delta_e)
    return - (jax.scipy.special.logsumexp(jnp.stack([log_l, log_r], axis=0), axis=0) / beta).sum()
  return _bistable_molecule


def sum_potentials(*components):
  def _summed(position, t, **kwargs):
    return sum([component(position, t, **kwargs)
                for component in components])
  return _summed


class BarrierCrossingSummary(NamedTuple):
  state: simulate.BrownianState
  work: jnp.array
  energy: jnp.array
  time: jnp.array


EnergyFn = Callable[[jnp.array, jnp.array], jnp.array]


def work_and_energy_fn(energy_fn: EnergyFn,
                       dt: Scalar
  ) -> Callable[[jnp.array, jnp.array], jnp.array]:
  # Computes the work done between `t - dt` and `t`.
  def _work_and_energy_fn(position, t):
    prev_nrg = energy_fn(position, t - dt)
    curr_nrg = energy_fn(position, t)
    return curr_nrg - prev_nrg, curr_nrg
  return _work_and_energy_fn


# TODO: unify the API for `simulate_barrier_crossing` and `simulate_ising`.

def simulate_barrier_crossing(energy_fn: EnergyFn,
                              temperature: Scalar,
                              gamma: Scalar,
                              total_time: Scalar,
                              time_steps: int,
                              shift_fn: Optional[space.ShiftFn] = None
  ) -> Callable[[jnp.array, jnp.array], BarrierCrossingSummary]:
  if shift_fn is None:
    _, shift_fn = space.free()
  dt = total_time / time_steps
  times = jnp.linspace(dt, total_time, time_steps)
  wrk_and_nrg = work_and_energy_fn(energy_fn, dt)

  init_fn, apply_fn = simulate.brownian(energy_fn, shift_fn, dt, temperature, gamma)

  def step(_state, t):
    wrk, nrg = wrk_and_nrg(_state.position, t)
    new_state = apply_fn(_state, t)
    return new_state, BarrierCrossingSummary(new_state, wrk, nrg, t)

  @jax.jit
  def _barrier_crossing(key, x0, mass=1.):
    state = init_fn(key, x0, mass)
    _, summary = jax.lax.scan(step, state, times)
    return summary

  return _barrier_crossing


# A `LossFn` maps (initial state, final_state, trajectory summary) to a scalar loss.
LossFn = Callable[[simulate.BrownianState, simulate.BrownianState, BarrierCrossingSummary], jnp.array]

# A `TrapFn` accepts a location function and returns an `EnergyFn` encoding the potential due to our trap.
TrapFn = Callable[[LocationFn], EnergyFn]


def total_work(initial_state: simulate.BrownianState,
               final_state: simulate.BrownianState,
               summary: BarrierCrossingSummary) -> jnp.array:
  del initial_state, final_state  # unused
  return summary.work.sum()


def estimate_gradient(trap_fn: TrapFn,
                      molecule: EnergyFn,
                      total_time: Scalar,
                      time_steps: int,
                      mass: Scalar,
                      temperature: Scalar,
                      gamma: Scalar,
                      shift_fn: Optional[space.ShiftFn] = None,
                      loss_fn: LossFn = total_work):
  @functools.partial(jax.grad, has_aux=True)
  def _estimate_gradient(location_schedule: LocationFn,
			 x0: jnp.array,
                         key: jnp.array):
    trap = trap_fn(location_schedule)
    energy_fn = sum_potentials(trap, molecule)
    simulate_crossing = simulate_barrier_crossing(energy_fn,
                                                  temperature,
                                                  gamma,
                                                  total_time,
                                                  time_steps,
                                                  shift_fn)
    # TODO: it is awkward that we compute initial state here _and_ inside `simulate`. Consider fixing this.
    initial_state = simulate.BrownianState(x0, mass, key, 0.)
    summary = simulate_crossing(key, x0, mass)
    final_state = map_slice(summary.state, -1)
    loss = loss_fn(initial_state, final_state, summary)
    log_prob = summary.state.log_prob.sum()
    gradient_estimator = log_prob * jax.lax.stop_gradient(loss) + loss
    return gradient_estimator, summary
  return _estimate_gradient

# A `TrainStepFn` takes (optimizer state, step, seed) and returns (new optimizer state, summary).
TrainStepFn = Callable[[jopt.OptimizerState, jnp.array, jnp.array],
                       Tuple[jopt.OptimizerState, BarrierCrossingSummary]]


def get_train_step(optimizer: jopt.Optimizer,
                   trap_fn: TrapFn,
                   molecule: EnergyFn,
                   total_time: Scalar,
                   time_steps: int,
                   mass: Scalar,
                   temperature: Scalar,
                   gamma: Scalar,
                   batch_size: int,
                   shift_fn: Optional[space.ShiftFn] = None,
                   loss_fn: LossFn = total_work,
  ) -> TrainStepFn:
  gradient_estimator = estimate_gradient(trap_fn, molecule, total_time, time_steps, mass,
                                         temperature, gamma, shift_fn, loss_fn)
  mapped_gradient_estimate = jax.vmap(gradient_estimator, [None, 0])
  @jax.jit
  def _train_step(opt_state, x0, step, key):
    keys = jax.random.split(key, batch_size)
    schedule = optimizer.params_fn(opt_state)
    grads, summary = mapped_gradient_estimate(schedule, x0, keys)
    mean_grad = jax.tree_map(lambda x: jnp.mean(x, 0), grads)
    opt_state = optimizer.update_fn(step, mean_grad, opt_state)
    return opt_state, summary
  return _train_step


