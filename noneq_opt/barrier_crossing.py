"""Code for running and optimizing Ising simulations."""
from typing import Callable, Union

import jax
import jax.numpy as jnp
from jax_md import space, energy

from tensorflow_probability.substrates import jax as tfp
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

def _get_location_fn(location: LocationFnOrConstant) -> LocationFn:
  if not callable(location):
    return lambda t: location
  return location


def potential(displacement_fn: space.DisplacementFn,
              location_fn: LocationFnOrConstant,
              k: Scalar = 1.
  ) -> PotentialFn:
  location_fn = _get_location_fn(location_fn)
  def _potential(position, t, **unused_kwargs):
    d = space.distance(displacement_fn(position, location_fn(t)))
    return energy.simple_spring(d, epsilon=k, length=0).sum()
  return _potential


def bistable_molecule(displacement_fn: space.DisplacementFn,
                      location_fn: LocationFnOrConstant,
                      k_l: Scalar,
                      k_r: Scalar,
                      delta_e: Scalar,
                      beta: Scalar
  ) -> PotentialFn:
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