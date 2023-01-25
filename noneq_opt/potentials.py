import time, collections, functools, os, pickle, tqdm, typing

import jax
import jax.numpy as jnp

from jax import jit, grad, vmap, value_and_grad, random, lax, ops
import dataclasses
from jax.example_libraries import optimizers as jopt

from jax_md import space, minimize, simulate, energy, quantity, smap, util
import jax_md as jmd
from jax_md import util 
static_cast = util.static_cast

f32 = jnp.float32
f64 = jnp.float64

import scipy as sp
import scipy.special as sps

from numpy import polynomial
import numpy as onp 

from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions

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

def V_spring(k_s):
  def total_energy(position, r0=0.0, **unused_kwargs):
      x = position[0][0]
      Es = k_s/2 * (x-r0) ** 2
      return Es
  return total_energy
