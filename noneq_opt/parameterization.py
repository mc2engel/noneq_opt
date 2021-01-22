"""Classes for parameterizing real valued functions [0, 1] -> R."""

import typing

import jax
import jax.numpy as jnp
import numpy as np
import scipy.special as sps


class Constant(typing.NamedTuple):
    value: jnp.array

    def __call__(self, x):
        return self.value * jnp.ones_like(self.value)


class PiecewiseLinear(typing.NamedTuple):
    values: jnp.array

    @property
    def length(self):
        return self.values.shape[-1]

    @property
    def x(self):
        return jnp.linspace(0, 1, self.length)

    def __call__(self, x):
        return jnp.interp(
        x, self.x, self.values, left=self.values[0], right=self.values[-1])


def chebyshev_coefficients(degree):
    # TODO: consider other Chebyshev polynomial type.
    return np.stack([
        np.concatenate([np.zeros(degree - j), sps.chebyt(j, True)])
        for j in range(degree + 1)])


class Chebyshev(typing.NamedTuple):
  weights: jnp.array

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
