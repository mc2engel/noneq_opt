import collections
from typing import Union, Callable

import jax
import jax.numpy as jnp
import jax_md as jmd
import distrax

from jax import random, jit, grad, vmap, value_and_grad, lax, ops

import time, collections, functools, pickle, typing, pdb
from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions

# Below is a custom Brownian simulator. It is identical to the one provided by JAX MD except that `BrownianState`
# is augmented with the log probability of the most recent step.

# Note that this implementation uses a different method for Gaussian samples, so it will produce identically
# _distributed_ values but not identical values to the JAX MD implementation.

# Furthermore, note that this implementation is designed for use with a REINFORCE approach to gradient estimation. In
# particular, we don't assume that we know the functional relationship between the position returned by `apply_fn` and
# the given `energy_or_force`.

Scalar = Union[jnp.array, float]
PotentialFn = Callable[[jnp.array, Scalar], jnp.array]
ShiftFn = Callable[[jnp.array, jnp.array], jnp.array]

class BrownianState(collections.namedtuple('BrownianState',
                                           'position mass rng log_prob')):
  pass

def brownian(energy_or_force,
             shift,
             dt,
             kT=0.1,
             gamma=0.1):
  """Simulation of Brownian dynamics.
  This code simulates Brownian dynamics which are synonymous with the overdamped
  regime of Langevin dynamics. However, in this case we don't need to take into
  account velocity information and the dynamics simplify. Consequently, when
  Brownian dynamics can be used they will be faster than Langevin. As in the
  case of Langevin dynamics our implementation follows [1].
  Args:
    energy_or_force: A function that produces either an energy or a force from
      a set of particle positions specified as an ndarray of shape
      [n, spatial_dimension].
    shift_fn: A function that displaces positions, R, by an amount dR. Both R
      and dR should be ndarrays of shape [n, spatial_dimension].
    dt: Floating point number specifying the timescale (step size) of the
      simulation.
    T_schedule: Either a floating point number specifying a constant temperature
      or a function specifying temperature as a function of time.
    quant: Either a quantity.Energy or a quantity.Force specifying whether
      energy_or_force is an energy or force respectively.
    gamma: A float specifying the friction coefficient between the particles
      and the solvent.
  Returns:
    See above.
    [1] E. Carlon, M. Laleman, S. Nomidis. "Molecular Dynamics Simulation."
        http://itf.fys.kuleuven.be/~enrico/Teaching/molecular_dynamics_2015.pdf
        Accessed on 06/05/2019.
  """

  force_fn = jmd.quantity.canonicalize_force(energy_or_force)

  #dt, gamma = jmd.util.static_cast(dt, gamma)

  kT = jmd.interpolate.canonicalize(kT)

  def _dist(state, t, **kwargs):
    nu = jnp.float32(1) / (state.mass * gamma)
    F = force_fn(state.position, t=t, **kwargs)
    mean =  F * nu * dt
    variance = jnp.float32(2) * kT(t) * dt * nu
    return tfd.Normal(mean, jnp.sqrt(variance))
  
  def init_fn(key, R, mass=jnp.float32(1)):
    #mass = jmd.quantity.canonicalize_mass(mass)
    return BrownianState(R, mass, key, 0.)

  def apply_fn(state, t=jnp.float32(0), **kwargs):
    dist = _dist(state, t, **kwargs)
    key, split = jax.random.split(state.rng)

    # We have to stop gradients here, otherwise the gradient with respect to
    # energy/force is zero. The following is a simple repro of the issue:
    #  def sample_log_prob(mean, key):
    #    d = tfd.Normal(mean, 1.)
    #    s = d.sample(seed=key)
    #    return d.log_prob(s)
    #  jax.grad(sample_log_prob)(0., key)  # Always 0 !
    dR = jax.lax.stop_gradient(dist.sample(seed=split))

    log_prob = dist.log_prob(dR).sum()
    R = shift(state.position, dR, t=t, **kwargs)
    return BrownianState(R, state.mass, key, log_prob)

  return init_fn, apply_fn

def run_brownian_stationary_trap(energy_fn, init_position, r0, shift, key, simulation_steps, dt, kT, gamma):
  key, split = random.split(key)  
  #init, apply = simulate.brownian(energy_fn, shift, dt=dt, kT=kT, gamma=mgamma)
  init, apply = brownian(energy_fn, shift, dt=dt, kT=kT, gamma=gamma)
  apply = jit(apply)

  state = init(split, init_position)
  @jit
  def scan_fn(state, step):
    # Dynamically pass r0 to apply, which passes it on to energy_fn
    state = apply(state, t=step, r0=r0)
    return state, state.position

  state, positions = lax.scan(scan_fn,state,(jnp.arange(simulation_steps)))
  return positions

def single_brownian_stiffness(energy_fn, init_position, r0_init, shift_fn, sim_length, dt=1e-6, kT=1e-5, mass=1.0, gamma=1.0):
  def _single_brownian_eqm(seed): 
      traj = run_brownian_stationary_trap(energy_fn, init_position, r0_init, shift_fn, seed, sim_length, dt, kT, gamma) 
      return traj
  return _single_brownian_eqm

def mapped_brownian_stiffness(batch_size, energy_fn, init_position, r0_init, shift_fn, sim_length, dt, kT, mass, gamma):
    mapped_eqm = jax.vmap(single_brownian_stiffness(energy_fn, init_position, r0_init, shift_fn, sim_length, dt=dt, kT=kT, mass=mass, gamma=gamma), [0])  
    @jax.jit
    def _mapped_brownian_eqm(seed):
      seeds = jax.random.split(seed, batch_size)
      boltz_dist = mapped_eqm(seeds)
      return boltz_dist
    return _mapped_brownian_eqm
