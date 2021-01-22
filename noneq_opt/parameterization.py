"""Classes for parameterizing real valued functions [0, 1] -> R."""

import typing

import jax
import jax.numpy as jnp
import numpy as np

class Constant(typing.NamedTuple):
    value: jnp.array

    def __call__(self, x):
        return self.value * jnp.ones_like(self.value)
