import time, collections, functools, os, pickle, typing, pdb
from tqdm import tqdm

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

import potentials
import simulate

import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.animation import ArtistAnimation
from matplotlib import rc
from matplotlib import colors
import seaborn as sns
rc('animation', html='jshtml')
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

 
N = 1
dim = 1
batch_size= 1
D = 2e6 #nm^2/s
kT = 4.183 #at 303K=30C in pN nm
beta=1.0/kT #
mgamma = kT/D
#trap params:
r0 = 0.
r0_final = 40.
k_s =1.0#pN/nm
init_position=0.*jnp.ones((N,dim)) #oxdna sim units
key = random.PRNGKey(int(time.time()))
key, split1 = random.split(key, 2)

energy_fn = potentials.V_spring(k_s)
force_fn = quantity.force(energy_fn)
displacement_fn, shift_fn = space.free()

dts = [1e-5, 1e-6]
sim_lengths = {1e-5: 100000, 1e-6: 1000000}
positions = []
for dt in dts:
  print("running for ", dt)
  eqm_fn = simulate.mapped_brownian_stiffness(batch_size, energy_fn, init_position, r0, shift_fn, sim_lengths[dt], dt, kT, mass=1.0, gamma=mgamma)
  key = random.PRNGKey(int(time.time()))
  key, split1 = random.split(key, 2)
  seed = jnp.zeros_like(split1)
  init_pos_batch = eqm_fn(seed)
  
  positions.append(jnp.reshape(init_pos_batch,(int(batch_size*sim_lengths[dt]),1)))

end_time = dts[0] * sim_lengths[dts[0]]
times = []
for dt in dts:
	times.append(jnp.linspace(0, end_time, sim_lengths[dt]))

for i in jnp.arange(len(times)):
	plt.plot(times[i], positions[i])
plt.show()
index=0
hists = []
bin_edges = [] 
for pos in positions:
	shaped_pos = jnp.reshape(pos,(int(batch_size*sim_lengths[dts[index]]),1))
	hist, bin_edges_ = jnp.histogram(shaped_pos, bins = 50)
	hists.append(hist)
	bin_edges.append(bin_edges_)
	index+=1
	plt.plot(bin_edges_[1:],hist)

