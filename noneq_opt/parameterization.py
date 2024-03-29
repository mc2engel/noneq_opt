"""This module provides classes for parameterized, real-valued functions."""
# TODO: type annotations.

import abc
import dataclasses
from typing import Callable

import jax
import jax_cosmo.scipy.interpolate as interpolate
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

NOTE: all of these classes work well for scalar -> scalar functions, but there is some inconsistency in how they
handle scalar -> vector mappings. TODO: fix this!
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
  value: np.ndarray

  @property
  def variables(self):
    return dict(value=self.value)

  @property
  def domain(self):
    return (0., 1.)

  def __call__(self, x):
    return self.value * jnp.ones_like(x)


class PiecewiseLinear(Parameterization):
  values: np.ndarray
  x0: np.ndarray = jnp.array(0.)
  x1: np.ndarray = jnp.array(1.)
  y0: np.ndarray = jnp.array(0.)
  y1: np.ndarray = jnp.array(1.)

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
  # Weights must have shape `[ndim, degree]`.
  weights: np.ndarray

  @property
  def variables(self):
    return dict(weights=self.weights)

  @property
  def domain(self):
    return (0., 1.)

  @property
  def degree(self):
    return self.weights.shape[-1] - 1

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
    x = 2 * x - 1  # Rescale [0, 1] -> [-1, 1].
    x_powers = self._powers(x)
    return jnp.einsum(
      '...w,wp,p...->...', self.weights, self.coefficients, x_powers)


class Spline(Parameterization):
  knots: np.ndarray
  values: np.ndarray
  # TODO(jamieas): Add option for free endpoints.
  x0: np.ndarray = jnp.array(0.)
  x1: np.ndarray = jnp.array(1.)
  y0: np.ndarray = jnp.array(0.)
  y1: np.ndarray = jnp.array(1.)
  spline_degree: int = 2
  spline_endpoints: str = 'not-a-knot'
  variable_knots: bool = False

  @property
  def variables(self):
    vars = dict(values=self.values)
    if self.variable_knots:
      vars['knots'] = self.knots
    return vars

  @property
  def constants(self):
    consts = dict(x0=self.x0,
                  x1=self.x1,
                  y0=self.y0,
                  y1=self.y1,
                  spline_degree=self.spline_degree,
                  spline_endpoints=self.spline_endpoints,
                  variable_knots=self.variable_knots)
    if not self.variable_knots:
      consts['knots'] = self.knots
    return consts

  @property
  def domain(self):
    return (self.x0, self.x1)

  @property
  def degree(self):
    return self.knots.shape[-1]

  @property
  def length(self):
    return self.degree + 2

  @property
  def x(self):
    return jnp.concatenate([
      self.x0[jnp.newaxis],
      self.knots,
      self.x1[jnp.newaxis]
    ])

  @property
  def y(self):
    return jnp.concatenate([
      self.y0[jnp.newaxis],
      self.values,
      self.y1[jnp.newaxis]
    ])

  @classmethod
  def equidistant(cls, d, x0=0., x1=1., y0=0., y1=1., **kwargs):
    x0 = jnp.array(x0)
    x1 = jnp.array(x1)
    y0 = jnp.array(y0)
    y1 = jnp.array(y1)
    knots = jnp.linspace(x0, x1, d + 2)[1:-1]
    values = jnp.linspace(y0, y1, d + 2)[1:-1]
    return cls(knots, values, x0, x1, y0, y1, **kwargs)

  @classmethod
  def from_baseline(cls, baseline, d, x0=0., x1=1., **kwargs):
    x0 = jnp.array(x0)
    x1 = jnp.array(x1)
    y0 = baseline(x0)
    y1 = baseline(x1)
    knots = jnp.linspace(x0, x1, d + 2)[1:-1]
    values = baseline(knots)
    return cls(knots, values, x0, x1, y0, y1, **kwargs)

  def __call__(self, x):
    spline = interpolate.InterpolatedUnivariateSpline(
      self.x, self.y, k=self.spline_degree, endpoints=self.spline_endpoints)
    return spline(x)


class ChangeDomain(Parameterization):
  """Wraps another `Parameterization`, linearly moving to a new domain."""
  wrapped: Parameterization
  x0: np.ndarray
  x1: np.ndarray

  @property
  def variables(self):
    return dict(wrapped=self.wrapped)

  @property
  def constants(self):
    return dict(
      x0=self.x0,
      x1=self.x1
    )

  @property
  def domain(self):
    return (self.x0, self.x1)

  def __call__(self, x):
    scale = (self.wrapped.domain[1] - self.wrapped.domain[0]) / (self.x1 - self.x0)
    _x = (x - self.x0) * scale + self.wrapped.domain[0]
    return self.wrapped(_x)


class ConstrainEndpoints(Parameterization):
  """Constrains the endpoints of `wrapped`.

  The resulting function will not resemble `wrapped` in general.
  """
  wrapped: Parameterization
  y0: np.ndarray
  y1: np.ndarray

  @property
  def variables(self):
    return dict(wrapped=self.wrapped)

  @property
  def constants(self):
    return dict(
    y0=self.y0,
    y1=self.y1
    )

  @property
  def domain(self):
    return self.wrapped.domain

  def __call__(self, x):
    x0, x1 = self.domain
    linear_component = self.y0 + (x - x0) / (x1 - x0) * (self.y1 - self.y0)
    return (x - x0) * (x1 - x) * self.wrapped(x) + linear_component


class AddBaseline(Parameterization):
  """Adds `baseline` to `wrapped`. If `baseline` is a `Parameterization`, it will not be fitted."""

  wrapped: Parameterization
  baseline: Callable

  @property
  def variables(self):
    return dict(wrapped=self.wrapped)

  @property
  def constants(self):
    return dict(baseline=self.baseline)

  @property
  def domain(self):
    return self.wrapped.domain

  def __call__(self, x):
    return self.wrapped(x) + self.baseline(x)

