"""This module provides classes for parameterized, real-valued functions."""
import abc
import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import scipy.special as sps

"""
### Parameterizations ###

Each of the classes below is a real-valued scalar function that is registered
as a JAX pytree. This makes them compatible with JAX transforms and allows
us to apply optimizer updates directly on the class.

The fields of each class are divided between `variables` and `constants`. The
pytree logic is such that JAX transformations such as `grad` or `vmap` will
apply only to the `variables` of each class. As an example, we might want a
piecewise linear function with fixed endpoints; the endpoints of each linear
component would be `variables`, but the endpoints of the entire function are
`constants`.
"""


class Parameterization(metaclass=abc.ABCMeta):

  # TODO: consider implementing `variables` and `constants` using `asdict`
  # and a list of names of the variables for each class.

  @property
  def variables(self):
    return dict()

  @property
  def constants(self):
    return dict()

  @abc.abstractmethod
  def domain(self):
    pass

  @abc.abstractmethod
  def __call__(self, x):
    pass

  def tree_flatten(self):
    return ((self.variables,), self.constants)

  @classmethod
  def tree_unflatten(cls, constants, variables):
    variables, = variables
    return cls(**variables, **constants)

  def __init_subclass__(cls):
    return jax.tree_util.register_pytree_node_class(dataclasses.dataclass(cls))


class Constant(Parameterization):
  value: jnp.array

  @property
  def variables(self):
    return dict(value=self.value)

  @property
  def domain(self):
    return (0., 1.)

  def __call__(self, x):
    return self.value * jnp.ones_like(x)


class PiecewiseLinear(Parameterization):
  values: jnp.array
  x0: jnp.array = jnp.array(0.)
  x1: jnp.array = jnp.array(1.)
  y0: jnp.array = jnp.array(0.)
  y1: jnp.array = jnp.array(1.)

  @property
  def variables(self):
    return dict(values=self.values)

  @property
  def constants(self):
    return dict(x0=self.x0,
                x1=self.x1,
                y0=self.y0,
                y1=self.y1)

  @property
  def domain(self):
    return (self.x0, self.x1)

  @property
  def degree(self):
    return self.values.shape[-1]

  @property
  def length(self):
    return self.degree + 2

  @property
  def x(self):
    return jnp.linspace(self.x0, self.x1, self.length)

  @property
  def y(self):
    return jnp.concatenate([self.y0[jnp.newaxis],
                            self.values,
                            self.y1[jnp.newaxis]])

  @classmethod
  def equidistant(cls, d, x0=0., x1=1., y0=0., y1=1.):
    x0 = jnp.array(x0)
    x1 = jnp.array(x1)
    y0 = jnp.array(y0)
    y1 = jnp.array(y1)
    values = jnp.linspace(y0, y1, d + 2)[1:-1]
    return cls(values, x0, x1, y0, y1)

  def __call__(self, x):
    return jnp.interp(
      x, self.x, self.y, left=self.y[0], right=self.y[-1])



def chebyshev_coefficients(degree):
  # TODO: consider other Chebyshev polynomial type.
  return np.stack([
    np.concatenate([np.zeros(degree - j), sps.chebyt(j, True)])
    for j in range(degree + 1)])


class Chebyshev(Parameterization):
  weights: jnp.array

  @property
  def variables(self):
    return dict(weights=self.weights)

  @property
  def domain(self):
    return (0., 1.)

  @property
  def degree(self):
    return self.weights.shape[0] - 1

  @property
  def coefficients(self):
    return chebyshev_coefficients(self.degree)

  def _powers(self, x):

    def _multiply_by_x(y, _):
      y *= x
      return y, y

    ones = jnp.ones_like(x)
    _, powers = jax.lax.scan(
      _multiply_by_x, ones, None, length=self.degree, reverse=True)
    return jnp.concatenate([powers, ones[jnp.newaxis]], axis=0)


  def __call__(self, x):
    x = 2. * x - 1.  # Rescale (0, 1) -> (-1, 1)
    x_powers = self._powers(x)
    return jnp.einsum(
      'w,wp,p...->...', self.weights, self.coefficients, x_powers)


class ChangeDomain(Parameterization):
  """Wraps another `Parameterization`, linearly moving to a new domain."""
  wrapped: Parameterization
  x0: jnp.array
  x1: jnp.array

  @property
  def variables(self):
    return dict(wrapped=self.wrapped)

  @property
  def domain(self):
    return (self.x0, self.x1)

  @property
  def constants(self):
    return dict(
      x0=self.x0,
      x1=self.x1
    )

  def __call__(self, x):
    scale = (self.wrapped.domain[1] - self.wrapped.domain[0]) / (self.x1 - self.x0)
    _x = (x - self.x0) * scale + self.wrapped.domain[0]
    return self.wrapped(_x)