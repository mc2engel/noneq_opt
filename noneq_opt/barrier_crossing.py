"""Code for running and optimizing Barrier Crossing simulations."""
import functools
import collections 

import jax
from jax import random,jit,grad,vmap,value_and_grad
import jax.numpy as jnp
from jax.example_libraries import optimizers as jopt
from jax_md import space, energy
from jax_md import simulate as reparameterize_simulate
import jax_md as jmd

import numpy as onp

from noneq_opt import simulate as reinforce_simulate
from noneq_opt import parameterization as p10n

import matplotlib.pyplot as plt

from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions


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


def run_brownian_opt(energy_fn, coeffs, init_position, r0_init, r0_final, Neq, shift, key, simulation_steps, dt=1e-5, kT=1e-5, mass=1.0, gamma=1.0):
  """Simulation of Brownian particle being dragged by a moving harmonic trap.
  Args:
    energy_fn: the function that governs particle energy. Here, an external harmonic potential
    r0_init: initial position of the trap, for which equilibration is done
    Neq: number of equilibration steps
    shift: shift_fn governing the Brownian simulation
    key: random key
    num_steps: total # simulation steps
    dt: integration time step
    temperature: simulation temperature kT

  Returns:
    total work required to drag the particle, from eq'n 17 in Jarzynski 2008
  """

  trap_fn = make_trap_fxn(jnp.arange(simulation_steps+1),coeffs,r0_init,r0_final)

  def equilibrate(init_state, Neq, apply, r0_init):
    @jit
    def scan_eq(state, step):
      state = apply(state, step, r0=r0_init)
      return state, 0
    state, _ = jax.lax.scan(scan_eq,init_state,jnp.arange(Neq))
    return state

  def increment_work(state, step):
        return (energy_fn(state.position, r0=trap_fn(step)) - energy_fn(state.position, r0=trap_fn(step-1)))

  @jit
  def scan_fn(state, step):
    dW = increment_work(state, step) #increment based on position BEFORE 'thermal kick' a la Crooks
    # Dynamically pass r0 to apply, which passes it on to energy_fn
    state = apply(state, step, r0=trap_fn(step))
    return state, (state.position, state.log_prob, dW)

  key, split = random.split(key)

  init, apply = brownian(energy_fn, shift, dt=dt, kT=kT, gamma=gamma)
  apply = jit(apply)

  eq_state = equilibrate(init(split, init_position, mass=mass), Neq, apply, r0_init)
  state = eq_state
  state, (positions, log_probs, works) = jax.lax.scan(scan_fn,state,(jnp.arange(simulation_steps)+1))
  works = jnp.concatenate([jnp.reshape(0., [1]), works])
  positions = jnp.concatenate([jnp.reshape(eq_state.position, [1,1,1]), positions])
  return positions, log_probs, works

def MakeSchedule_chebyshev(timevec,coeffs,r0_init,r0_final):
  timevec = timevec[1:-1]
  scaled_timevec = (timevec-1)/(timevec[-1]-1)
  vals = p10n.Chebyshev(coeffs)(scaled_timevec)
  vals = jnp.concatenate([vals, jnp.reshape(r0_final, [1])])
  vals = jnp.concatenate([jnp.reshape(r0_init, [1]), vals])
  return vals

def make_trap_fxn(timevec,coeffs,r0_init,r0_final):
  positions = MakeSchedule_chebyshev(timevec,coeffs,r0_init,r0_final)
  def Get_r0(step):
    return positions[step]
  return Get_r0

def theoretical_opt(t, tvec, lambda_t):
    return jnp.interp(t, tvec, lambda_t)

def make_theoretical_trap_fxn(sim_stepvec, dt, tvec, lambda_t):
    positions = theoretical_opt(sim_stepvec*dt, tvec, lambda_t)
    def Get_r0(step):
        return positions[step]
    return Get_r0

def fit_linear_to_cheby(end_time, dt, r0_init, r0_final,degree):
    simulation_steps = int((end_time)/dt)
    slope = (r0_final - r0_init)/(simulation_steps)
    vals = slope*onp.arange(1,simulation_steps)+r0_init
    xscaled=(onp.arange(1,simulation_steps)-1)/(simulation_steps-2)
    p = onp.polynomial.Chebyshev.fit(xscaled, vals, degree, domain=[0,1])
    return p10n.Chebyshev(p.coef).weights

def V_biomolecule(kappa_l, kappa_r, x_m, delta_E, k_s, beta):
  def total_energy(position, r0=0.0, **unused_kwargs):
      x = position[0][0]
      #underlying energy landscape:
      Em = -(1./beta)*jnp.log(jnp.exp(-0.5*beta*kappa_l*(x+x_m)**2)+jnp.exp(-(0.5*beta*kappa_r*(x-x_m)**2+beta*delta_E)))
      #moving harmonic potential:
      Es = k_s/2 * (x-r0) ** 2
      return Em + Es
  return total_energy

def V_biomolecule_shifted(kappa_l, kappa_r, x_m, delta_E, k_s, beta):
  def total_energy(position, r0=0.0, **unused_kwargs):
      x = position[0][0]
      #underlying energy landscape:
      Em = -(1./beta)*jnp.log(jnp.exp(-0.5*beta*kappa_l*(x)**2)+jnp.exp(-(0.5*beta*kappa_r*(x-2.*x_m)**2+beta*delta_E)))
      #moving harmonic potential:
      Es = k_s/2 * (x-r0) ** 2
      return Em + Es
  return total_energy

def plot_with_stddev(x, label=None, n=1, axis=0, ax=plt):
  stddev = jnp.std(x, axis)
  mn = jnp.mean(x, axis)

  ax.fill_between(jnp.arange(mn.shape[0]),
                  mn + n * stddev, mn - n * stddev, alpha=.3)
  ax.plot(mn, '-o', label=label)

def single_estimate_boltzinit(energy_fn, r0_init, r0_final, Neq, shift, simulation_steps, dt, temperature, mass, gamma):
  @functools.partial(jax.value_and_grad, has_aux=True)#the 'aux' is the summary
  def _single_estimate(coeffs, init_pos, seed): #function only of the params to be differentiated w.r.t.
      positions, log_probs, works = run_brownian_opt(energy_fn, coeffs, init_pos, r0_init, r0_final, Neq, shift, seed, simulation_steps, dt, temperature, mass, gamma)
      total_work = works.sum()
      tot_log_prob = log_probs.sum()
      summary = (positions, tot_log_prob, total_work)
      #summary = (positions, tot_log_prob, works)
      gradient_estimator = (tot_log_prob * jax.lax.stop_gradient(total_work) + total_work)
      return gradient_estimator, summary
  return _single_estimate

def estimate_gradient_boltzinit(batch_size, energy_fn, r0_init, r0_final, Neq, shift, simulation_steps, dt, temperature, mass, gamma):
    mapped_estimate = jax.vmap(single_estimate_boltzinit(energy_fn, r0_init, r0_final, Neq, shift, simulation_steps, dt, temperature, mass, gamma), [None, 0, 0])
    #mapped_estimate = jax.soft_pmap(lambda s: single_estimate(energy_fn, init_position, r0_init, r0_final, Neq, shift, simulation_steps, dt, temperature, mass, gamma), [None, 0])
    @jax.jit #why am I allowed to jit something whose output is a function? Thought it had to be pytree output...
    def _estimate_gradient(coeffs, init_pos, seed):
      seeds = jax.random.split(seed, batch_size)
      (gradient_estimator, summary), grad = mapped_estimate(coeffs, init_pos, seeds)
      return jnp.mean(grad, axis=0), (gradient_estimator, summary)
    return _estimate_gradient


def run_brownian_stationary_trap(energy_fn, init_position, r0, shift, key, simulation_steps, dt, kT, mgamma):
  key, split = random.split(key)
  #init, apply = simulate.brownian(energy_fn, shift, dt=dt, kT=kT, gamma=mgamma)
  init, apply = brownian(energy_fn, shift, dt=dt, kT=kT, gamma=mgamma)
  apply = jit(apply)

  state = init(split, init_position)
  @jit
  def scan_fn(state, step):
    # Dynamically pass r0 to apply, which passes it on to energy_fn
    state = apply(state, t=step, r0=r0)
    return state, state.position

  state, positions = jax.lax.scan(scan_fn,state,(jnp.arange(simulation_steps)))
  return positions

def single_brownian_eqm(energy_fn, init_position, r0_init, shift_fn, sim_length, dt=1e-6, kT=1e-5, mass=1.0, gamma=1.0):
  def _single_brownian_eqm(seed):
      traj = run_brownian_stationary_trap(energy_fn, init_position, r0_init, shift_fn, seed, sim_length, dt, kT, gamma)
      return traj[-1]
  return _single_brownian_eqm

def mapped_brownian_eqm(batch_size, energy_fn, init_position, r0_init, shift_fn, sim_length, dt=1e-6, kT=1e-5, mass=1.0, gamma=1.0):
    mapped_eqm = jax.vmap(single_brownian_eqm(energy_fn, init_position, r0_init, shift_fn, sim_length, dt=1e-6, kT=1e-5, mass=1.0, gamma=1.0), [0])
    @jax.jit
    def _mapped_brownian_eqm(seed):
      seeds = jax.random.split(seed, batch_size)
      boltz_dist = mapped_eqm(seeds)
      return boltz_dist
    return _mapped_brownian_eqm



###FUNCTIONS FOR EXPLORING THE REINFORCE GRADIENT####

def single_estimate_boltzinit_logP(energy_fn, r0_init, r0_final, Neq, shift, simulation_steps, dt, temperature, mass, gamma):
  @functools.partial(jax.value_and_grad, has_aux=True)#the 'aux' is the summary
  def _single_estimate(coeffs, init_pos, seed): #function only of the params to be differentiated w.r.t.
      positions, log_probs, works = run_brownian_opt(energy_fn, coeffs, init_pos, r0_init, r0_final, Neq, shift, seed, simulation_steps, dt, temperature, mass, gamma)
      total_work = works.sum()
      tot_log_prob = log_probs.sum()
      summary = (positions, tot_log_prob, total_work)
      #summary = (positions, tot_log_prob, works)
      grad_logP_term = tot_log_prob * jax.lax.stop_gradient(total_work)
      grad_W_term = total_work
      gradient_estimator = grad_logP_term + grad_W_term
      #gradient_estimator = (tot_log_prob * jax.lax.stop_gradient(total_work) + total_work)
      return grad_logP_term, summary
  return _single_estimate

def estimate_gradient_boltzinit_logP(batch_size, energy_fn, r0_init, r0_final, Neq, shift, simulation_steps, dt, temperature, mass, gamma):
    mapped_estimate = jax.vmap(single_estimate_boltzinit_logP(energy_fn, r0_init, r0_final, Neq, shift, simulation_steps, dt, temperature, mass, gamma), [None, 0, 0])
    #mapped_estimate = jax.soft_pmap(lambda s: single_estimate(energy_fn, init_position, r0_init, r0_final, Neq, shift, simulation_steps, dt, temperature, mass, gamma), [None, 0])
    @jax.jit #why am I allowed to jit something whose output is a function? Thought it had to be pytree output...
    def _estimate_gradient(coeffs, init_pos, seed):
      seeds = jax.random.split(seed, batch_size)
      (grad_logP_term, summary), grad = mapped_estimate(coeffs, init_pos, seeds)
      return jnp.mean(grad, axis=0), (grad_logP_term, summary)
    return _estimate_gradient


def single_estimate_boltzinit_reparam(energy_fn, r0_init, r0_final, Neq, shift, simulation_steps, dt, temperature, mass, gamma):
  @functools.partial(jax.value_and_grad, has_aux=True)#the 'aux' is the summary
  def _single_estimate(coeffs, init_pos, seed): #function only of the params to be differentiated w.r.t.
      positions, log_probs, works = run_brownian_opt(energy_fn, coeffs, init_pos, r0_init, r0_final, Neq, shift, seed, simulation_steps, dt, temperature, mass, gamma)
      total_work = works.sum()
      tot_log_prob = log_probs.sum()
      summary = (positions, tot_log_prob, total_work)
      #summary = (positions, tot_log_prob, works)
      grad_logP_term = tot_log_prob * jax.lax.stop_gradient(total_work)
      grad_W_term = total_work
      gradient_estimator = grad_logP_term + grad_W_term
      #gradient_estimator = (tot_log_prob * jax.lax.stop_gradient(total_work) + total_work)
      return grad_W_term, summary
  return _single_estimate

def estimate_gradient_boltzinit_reparam(batch_size, energy_fn, r0_init, r0_final, Neq, shift, simulation_steps, dt, temperature, mass, gamma):
    mapped_estimate = jax.vmap(single_estimate_boltzinit_reparam(energy_fn, r0_init, r0_final, Neq, shift, simulation_steps, dt, temperature, mass, gamma), [None, 0, 0])
    #mapped_estimate = jax.soft_pmap(lambda s: single_estimate(energy_fn, init_position, r0_init, r0_final, Neq, shift, simulation_steps, dt, temperature, mass, gamma), [None, 0])
    @jax.jit #why am I allowed to jit something whose output is a function? Thought it had to be pytree output...
    def _estimate_gradient(coeffs, init_pos, seed):
      seeds = jax.random.split(seed, batch_size)
      (grad_W_term, summary), grad = mapped_estimate(coeffs, init_pos, seeds)
      return jnp.mean(grad, axis=0), (grad_W_term, summary)
    return _estimate_gradient

def single_estimate_boltzinit_REINFORCE_recordall(energy_fn, r0_init, r0_final, Neq, shift, simulation_steps, dt, temperature, mass, gamma):
  @functools.partial(jax.value_and_grad, has_aux=True)#the 'aux' is the summary
  def _single_estimate(coeffs, init_pos, seed): #function only of the params to be differentiated w.r.t.
      positions, log_probs, works = run_brownian_opt(energy_fn, coeffs, init_pos, r0_init, r0_final, Neq, shift, seed, simulation_steps, dt, temperature, mass, gamma)
      total_work = works.sum()
      tot_log_prob = log_probs.sum()
      summary = (positions, tot_log_prob, total_work)
      #summary = (positions, tot_log_prob, works)
      gradient_estimator = (tot_log_prob * jax.lax.stop_gradient(total_work) + total_work)
      return gradient_estimator, summary
  return _single_estimate

def estimate_gradient_boltzinit_REINFORCE_recordall(batch_size, energy_fn, r0_init, r0_final, Neq, shift, simulation_steps, dt, temperature, mass, gamma):
    mapped_estimate = jax.vmap(single_estimate_boltzinit_REINFORCE_recordall(energy_fn, r0_init, r0_final, Neq, shift, simulation_steps, dt, temperature, mass, gamma), [None, 0, 0])
    #mapped_estimate = jax.soft_pmap(lambda s: single_estimate(energy_fn, init_position, r0_init, r0_final, Neq, shift, simulation_steps, dt, temperature, mass, gamma), [None, 0])
    @jax.jit #why am I allowed to jit something whose output is a function? Thought it had to be pytree output...
    def _estimate_gradient(coeffs, init_pos, seed):
      seeds = jax.random.split(seed, batch_size)
      (gradient_estimator, summary), grad = mapped_estimate(coeffs, init_pos, seeds)
      return jnp.mean(grad, axis=0), (gradient_estimator, summary)
    return _estimate_gradient
