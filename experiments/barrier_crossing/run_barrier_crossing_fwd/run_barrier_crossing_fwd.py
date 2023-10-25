import time
import collections
import functools
import os
import pickle

import jax
import jax.numpy as jnp

from jax.scipy import ndimage
from scipy import integrate

from jax import jit
from jax import grad
from jax import vmap
from jax import value_and_grad

from jax import random
from jax import lax
from jax import ops
import dataclasses

#from jax.experimental import stax
from jax.example_libraries import optimizers as jopt

#from jax.config import config
#config.update('jax_enable_x64', True)

from jax_md import space
from jax_md import minimize
from jax_md import simulate
from jax_md import space
from jax_md import energy
from jax_md import quantity
from jax_md import smap
import jax_md as jmd
from jax_md import util
static_cast = util.static_cast

f32 = jnp.float32
f64 = jnp.float64

import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.animation import ArtistAnimation
from matplotlib import rc
from matplotlib import colors
import seaborn as sns
rc('animation', html='jshtml')
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import tqdm
import time
import typing

import scipy as sp
import scipy.special as sps

from numpy import polynomial
import numpy as onp

from noneq_opt import barrier_crossing

end_time = float(sys.argv[1]) #128
dt = float(sys.argv[2]) #11 
#degree of Chebyshev polynomials that describe protocols
degree = int(sys.argv[3]) #32
kappa = float(sys.argv[4])
batch_size = int(sys.argv[5]) #256
savestring = str(sys.argv[6])

N = 1
dim = 1
simulation_steps = int((end_time)/dt)
D = 2e6 #nm^2/s
kT = 4.183 #at 303K=30C in pN nm
beta=1.0/kT #
mgamma = kT/D
#trap params:
r0_init = 0
r0_final = 40.
degree = 12
init_coeffs = barrier_crossing.fit_linear_to_cheby(end_time, dt, r0_init, r0_final, degree) 
Neq = 0 #num equilibration steps to perform within Brownian sim loop


#landscape params:
x_m=14. #nm #25/2 == rough contour length of unfolded 44 bp hairpin
delta_E=0 #
#kappa_l=21.3863/(beta*x_m**2)
kappa_l = kappa/(beta*x_m**2)
kappa_r=kappa_l #pN/nm 
k_trap = 0.4 # pN/nm 
init_position=r0_init*jnp.ones((N,dim))
key = random.PRNGKey(int(time.time()))
key, split1 = random.split(key, 2)

displacement_fn, shift_fn = space.free()

#equilibration in boltzmann dist:
dt_eq = 1e-6
eqm_sim_length = 50 # this should be approximately the correlation length. Since these sims are Brownian, the particle has no "memory" of where it has been... if
#inertial effects were at play, we would have to equilibrate for longer

energy_fn = barrier_crossing.V_biomolecule_shifted(kappa_l, kappa_r, x_m, delta_E, k_trap, beta)
force_fn = quantity.force(energy_fn)
eqm_fn = barrier_crossing.mapped_brownian_eqm(batch_size, energy_fn, init_position, r0_init, shift_fn, eqm_sim_length, dt_eq, kT, mass=1.0, gamma=mgamma)

#generate #batch_size samples from initial Boltzmann distribution.
split1, splitoon, splitoonoon = random.split(split1, 3)
init_pos_batch = eqm_fn(splitoon)
  
def make_batched_sim(energy_fn, coeffs, r0_init, r0_final, shift, simulation_steps, dt, kT, mass, gamma):
    def batched_sim(init_pos, seed):
        positions, log_probs, works = barrier_crossing.run_brownian_opt(energy_fn, coeffs, init_pos, r0_init, r0_final, Neq, shift, seed, simulation_steps, dt, kT, mass, gamma)
        return positions, log_probs, works
    return batched_sim

batched_sim = vmap(make_batched_sim(energy_fn, init_coeffs, r0_init, r0_final, shift_fn, simulation_steps, dt, kT, mass=1.0, gamma=mgamma), [0,0])

seeds = jax.random.split(splitoonoon, batch_size)
all_pos, all_log_probs, all_works = batched_sim(init_pos_batch, seeds)

print("average work: ", all_works.shape)


###SAVE OUTPUT###
#save output:
savedir = 'output/'
prefix = 'end_time_%i_dt_%i_kappa_%.3f_batch_%i_'%(end_time, dt,kappa_l,batch_size)

if not os.path.isdir(savedir):
    os.mkdir(savedir)

#afile = open(savedir+prefix+'summaries.pkl', 'wb')
#pickle.dump(summaries, afile)
#afile.close()

afile = open(savedir+prefix+savestring+'all_pos.pkl', 'wb')
pickle.dump(all_pos, afile)
afile.close()

afile = open(savedir+prefix+savestring+'all_works.pkl', 'wb')
pickle.dump(all_works, afile)
afile.close()

afile = open(savedir+prefix+savestring+'all_log_probs.pkl', 'wb')
pickle.dump(all_log_probs, afile)
afile.close()

