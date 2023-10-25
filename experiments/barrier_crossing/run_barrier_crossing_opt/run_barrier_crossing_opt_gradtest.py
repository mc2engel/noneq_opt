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

from noneq_opt import barrier_crossing

from numpy import polynomial
import numpy as onp
import sys

#parameters to plot:
end_time = float(sys.argv[1]) #128
dt = float(sys.argv[2]) #11 
#degree of Chebyshev polynomials that describe protocols
degree = int(sys.argv[3]) #32
kappa = float(sys.argv[4])
batch_size = int(sys.argv[5]) #256
opt_steps = int(sys.argv[6]) #10 

N = 1
dim = 1
simulation_steps = int((end_time)/dt)
D = 2e6 #nm^2/s
kT = 4.183 #at 303K=30C in pN nm
beta=1.0/kT #
mgamma = kT/D
#trap params:
r0_init = -10.
r0_final = 10.
init_coeffs = barrier_crossing.fit_linear_to_cheby(end_time, dt, r0_init, r0_final,degree) 
Neq = 0

#landscape params:
x_m=10. #nm #25/2 == rough contour length of unfolded 44 bp hairpin
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


energy_fn = barrier_crossing.V_biomolecule(kappa_l, kappa_r, x_m, delta_E, k_trap, beta)
force_fn = quantity.force(energy_fn)
eqm_fn = barrier_crossing.mapped_brownian_eqm(batch_size, energy_fn, init_position, r0_init, shift_fn, eqm_sim_length, dt_eq, kT, mass=1.0, gamma=mgamma)
lr = jopt.exponential_decay(0.1, opt_steps, 0.001)
optimizer = jopt.adam(lr)

coeffs_REINFORCE = []
coeffs_logP = []
coeffs_reparam = []
all_works_REINFORCE = []
all_works_logP = []
all_works_reparam = []
all_log_probs_REINFORCE = []
all_log_probs_logP = []
all_log_probs_reparam = []
grads_REINFORCE =[]
grads_logP = []
grads_logP_vals = [] #use the REINFORCE to propagate the optimization, but evaluate the size of each term at each step
grads_reparam_vals = [] #use the REINFORCE to propagate the optimization, but evaluate the size of each term at each step
grads_reparam = []


#save output:
savedir = 'output/'
prefix = 'end_time_%.10f_dt_%.10f_chebdeg_%i_kappa_%.4f_batch_%i_trainsteps_%i_' %(end_time, dt,degree,kappa,batch_size,opt_steps)

if not os.path.isdir(savedir):
    os.mkdir(savedir)


###full REINFORCE grad###
init_state = optimizer.init_fn(init_coeffs)
opt_state = optimizer.init_fn(init_coeffs)
grad_fxn = barrier_crossing.estimate_gradient_boltzinit(batch_size, energy_fn, r0_init, r0_final, Neq, shift_fn, simulation_steps, dt, kT, 1.0, mgamma)
grad_fxn_logP = barrier_crossing.estimate_gradient_boltzinit_logP(batch_size, energy_fn, r0_init, r0_final, Neq, shift_fn, simulation_steps, dt, kT, 1.0, mgamma)
grad_fxn_reparam = barrier_crossing.estimate_gradient_boltzinit_reparam(batch_size, energy_fn, r0_init, r0_final, Neq, shift_fn, simulation_steps, dt, kT, 1.0, mgamma)
coeffs_REINFORCE.append((0,) + (optimizer.params_fn(opt_state),))
#generate #batch_size samples from initial Boltzmann distribution. Use 1000 steps (teq=0.001 and dt=1e-6) to equilibrate in trap

for j in tqdm.trange(opt_steps,position=0):
  key = random.PRNGKey(int(time.time())) 
  key, splitoon, splitoonoon = random.split(key, 3)
  init_pos_batch = eqm_fn(splitoon)
  #run optimization on #batch_size sims:
  grad, (_, summary) = grad_fxn(optimizer.params_fn(opt_state), init_pos_batch, splitoonoon)
  grad_logP_val, (_, _) = grad_fxn_logP(optimizer.params_fn(opt_state), init_pos_batch, splitoonoon)
  grad_reparam_val, (_, _) = grad_fxn_reparam(optimizer.params_fn(opt_state), init_pos_batch, splitoonoon)
  opt_state = optimizer.update_fn(j, grad, opt_state)
  all_works_REINFORCE.append(summary[2])
  grads_REINFORCE.append(grad)
  grads_logP_vals.append(grad_logP_val)
  grads_reparam_vals.append(grad_reparam_val)
  if (j % 100 == 0):
    print("saving for j is ",j)
    #print(((j+1),) + (optimizer.params_fn(opt_state),))
    coeffs_REINFORCE.append(((j+1),) + (optimizer.params_fn(opt_state),));

coeffs_REINFORCE.append((opt_steps,) + (optimizer.params_fn(opt_state),))

afile = open(savedir+prefix+'all_works_REINFORCE.pkl', 'wb')
pickle.dump(all_works_REINFORCE, afile)
afile.close()

afile = open(savedir+prefix+'coeffs_REINFORCE.pkl', 'wb')
pickle.dump(coeffs_REINFORCE, afile)
afile.close()

afile = open(savedir+prefix+'grads_REINFORCE.pkl', 'wb')
pickle.dump(grads_REINFORCE, afile)
afile.close()

afile = open(savedir+prefix+'grads_REINFORCE_logP_vals.pkl', 'wb')
pickle.dump(grads_logP_vals, afile)
afile.close()

afile = open(savedir+prefix+'grads_REINFORCE_reparam_vals.pkl', 'wb')
pickle.dump(grads_reparam_vals, afile)
afile.close()

###log prob term of grad###
init_state = optimizer.init_fn(init_coeffs)
opt_state = optimizer.init_fn(init_coeffs)
grad_fxn = barrier_crossing.estimate_gradient_boltzinit_logP(batch_size, energy_fn, r0_init, r0_final, Neq, shift_fn, simulation_steps, dt, kT, 1.0, mgamma)
coeffs_logP.append((0,) + (optimizer.params_fn(opt_state),))
#generate #batch_size samples from initial Boltzmann distribution. Use 1000 steps (teq=0.001 and dt=1e-6) to equilibrate in trap

for j in tqdm.trange(opt_steps,position=0):
  key = random.PRNGKey(int(time.time())) 
  key, splitoon, splitoonoon = random.split(key, 3)
  init_pos_batch = eqm_fn(splitoon)
  #run optimization on #batch_size sims:
  grad, (_, summary) = grad_fxn(optimizer.params_fn(opt_state), init_pos_batch, splitoonoon)
  opt_state = optimizer.update_fn(j, grad, opt_state)
  all_works_logP.append(summary[2])
  grads_logP.append(grad)
  if (j % 100 == 0):
    #print(((j+1),) + (optimizer.params_fn(opt_state),))
    coeffs_logP.append(((j+1),) + (optimizer.params_fn(opt_state),));

coeffs_logP.append((opt_steps,) + (optimizer.params_fn(opt_state),))

afile = open(savedir+prefix+'all_works_logP.pkl', 'wb')
pickle.dump(all_works_logP, afile)
afile.close()

afile = open(savedir+prefix+'coeffs_logP.pkl', 'wb')
pickle.dump(coeffs_logP, afile)
afile.close()

afile = open(savedir+prefix+'grads_logP.pkl', 'wb')
pickle.dump(grads_logP, afile)
afile.close()

###reparameterization term of grad###
init_state = optimizer.init_fn(init_coeffs)
opt_state = optimizer.init_fn(init_coeffs)
grad_fxn = barrier_crossing.estimate_gradient_boltzinit_reparam(batch_size, energy_fn, r0_init, r0_final, Neq, shift_fn, simulation_steps, dt, kT, 1.0, mgamma)
coeffs_reparam.append((0,) + (optimizer.params_fn(opt_state),))
#generate #batch_size samples from initial Boltzmann distribution. Use 1000 steps (teq=0.001 and dt=1e-6) to equilibrate in trap

for j in tqdm.trange(opt_steps,position=0):
  key = random.PRNGKey(int(time.time())) 
  key, splitoon, splitoonoon = random.split(key, 3)
  init_pos_batch = eqm_fn(splitoon)
  #run optimization on #batch_size sims:
  grad, (_, summary) = grad_fxn(optimizer.params_fn(opt_state), init_pos_batch, splitoonoon)
  opt_state = optimizer.update_fn(j, grad, opt_state)
  all_works_reparam.append(summary[2])
  grads_reparam.append(grad)
  if (j % 100 == 0):
    #print(((j+1),) + (optimizer.params_fn(opt_state),))
    coeffs_reparam.append(((j+1),) + (optimizer.params_fn(opt_state),));

coeffs_reparam.append((opt_steps,) + (optimizer.params_fn(opt_state),))
afile = open(savedir+prefix+'all_works_reparam.pkl', 'wb')
pickle.dump(all_works_reparam, afile)
afile.close()

afile = open(savedir+prefix+'coeffs_reparam.pkl', 'wb')
pickle.dump(coeffs_reparam, afile)
afile.close()

afile = open(savedir+prefix+'grads_reparam.pkl', 'wb')
pickle.dump(grads_reparam, afile)
afile.close()


