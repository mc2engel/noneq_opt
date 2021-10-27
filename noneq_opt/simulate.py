import collections
from typing import Union, Callable

import jax
import jax.numpy as jnp
import jax_md as jmd
import distrax


# Below is a custom Brownian simulator. It is identical to the one provided by JAX MD except that `BrownianState`
# is augmented with the log probability of the most recent step.

# Note that this implementation uses a different method for Gaussian samples, so it will produce identically
# _distributed_ values but not identical values to the JAX MD implementation.

# Furthermore, note that this implementation is designed for use with a REINFORCE approach to gradient estimation. In
# particular, we don't assume that we know the functional relationship between the position returned by `apply_fn` and
# the given `energy_or_force`.

class BrownianState(collections.namedtuple('BrownianState',
                                           'position mass rng log_prob')):
  pass

Scalar = Union[jnp.array, float]
PotentialFn = Callable[[jnp.array, Scalar], jnp.array]
ShiftFn = Callable[[jnp.array, jnp.array], jnp.array]


def brownian(energy_or_force: PotentialFn,
             shift: ShiftFn,
             dt: jnp.array,
             T_schedule: Scalar,
             gamma: Scalar = 0.1):
  """Simulation of Brownian dynamics.
  This code simulates Brownian dynamics which are synonymous with the overdamped
  regime of Langevin dynamics. However, in this case we don't need to take into
  account velocity information and the dynamics simplify. Consequently, when
  Brownian dynamics can be used they will be faster than Langevin. As in the
  case of Langevin dynamics our implementation follows [1].
  Args:
    energy_or_force: A function that produces either an energy or a force from
      a set of particle positions specified as an ndarray of shape
      [n, spatial_dimension], along with a time.
    shift: A function that displaces positions, R, by an amount dR. Both R
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

  dt, gamma = jmd.util.static_cast(dt, gamma)

  T_schedule = jmd.interpolate.canonicalize(T_schedule)

  def init_fn(key, R, mass=jnp.float32(1)):
    mass = jmd.quantity.canonicalize_mass(mass)
    return BrownianState(R, mass, key, 0.)

  def _dist(state, t, **kwargs):
    nu = jnp.float32(1) / (state.mass * gamma)
    F = force_fn(state.position, t=t, **kwargs)
    mean = F * nu * dt
    variance = jnp.float32(2) * T_schedule(t) * dt * nu
    return distrax.Normal(mean, jnp.sqrt(variance))

  def apply_fn(state, t=jnp.float32(0), **kwargs):
    dist = _dist(state, t, **kwargs)
    key, split = jax.random.split(state.rng)

    # We have to stop gradients here, otherwise the gradient with respect to
    # energy/force is zero. The following is a simple repro of the issue:
    #  def sample_log_prob(mean, key):
    #    d = distrax.Normal(mean, 1.)
    #    s = d.sample(seed=key)
    #    return d.log_prob(s)
    #  jax.grad(sample_log_prob)(0., key)  # Always 0 !
    dR = jax.lax.stop_gradient(dist.sample(seed=split))

    log_prob = dist.log_prob(dR).sum(-1)
    R = shift(state.position, dR, t=t, **kwargs)
    return BrownianState(R, state.mass, key, log_prob)

  return init_fn, apply_fn