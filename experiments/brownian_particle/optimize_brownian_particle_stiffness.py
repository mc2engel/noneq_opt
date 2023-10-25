import functools
import numpy as onp

import jax
import jax.numpy as jnp

from jax.scipy import ndimage

from jax import jit, grad, vmap, value_and_grad, random, lax, ops

from jax.example_libraries import optimizers as jopt

from jax_md import space, minimize, simulate, energy, quantity, smap

f32 = jnp.float32
f64 = jnp.float64

import matplotlib.pyplot as plt
from matplotlib import colors
import seaborn as sns
import pandas as pd
import pickle
import csv
import sys
import tqdm
import time
import pdb

from noneq_opt import brownian_particle as bp

N = 1
dim = 1
end_time = float(sys.argv[1])#0.5
dt = 1e-3 #this needs to be sufficiently small.. check for strange behaviour!
simulation_steps = int((end_time)/dt)+1 #{0, dt, 2dt, ..., end time} add 1 for t=0
opt_steps = int(sys.argv[2])#100
batch_size= int(sys.argv[3])#1e5
temperature = 1. #kBT
key = random.PRNGKey(0)
key, split = random.split(key)

init_position = 15.*jnp.ones((N,dim)) #particle initial position
k_init = float(sys.argv[4])#0.2
k_final = float(sys.argv[5])#1.
trap_r0 = 15.
teq=10.
Neq=int(teq/dt)

lr = jopt.exponential_decay(0.1,opt_steps,0.001)
optimizer = jopt.adam(lr)

energy_fn = bp.Vspring_stiff(trap_r0)
force_fn = quantity.force(energy_fn)
displacement_fn, shift_fn = space.free()

init_schedule = jnp.linspace(k_init, k_final, 8)

summaries = []
schedules = []
all_works = []
opt_state = optimizer.init_fn(init_schedule)

grad_fxn = bp.estimate_gradient_stiffness(batch_size, energy_fn, k_init, k_final, Neq, shift_fn, simulation_steps, dt, temperature,init_position)

for j in tqdm.trange(opt_steps,position=0):
  key, split = random.split(key)
  grad, (_, summary) = grad_fxn(optimizer.params_fn(opt_state), split)
  opt_state = optimizer.update_fn(j, grad, opt_state)
  all_works.append(summary[2])
  #summaries.append(summary)
  #if j % (opt_steps // 10) == 0:
  schedules.append((j,) + (optimizer.params_fn(opt_state),))

print("final parameters: ", optimizer.params_fn(opt_state))
#all_summaries = jax.tree_multimap(lambda *args: jnp.stack(args), *summaries)
all_works = jax.tree_map(lambda *args: jnp.stack(args), *all_works)

savedir = '/n/brenner_lab/User/mengel/PROJECTS/NONEQ_REFACTOR_2023/noneq_opt/experiments/brownian_particle/output/'
prefix = 'stiffness_particle_simtime_%.3f_optsteps_%i_batch_%i_kinit_%.3f_kfinal_%.3f'%(end_time,opt_steps,batch_size,k_init,k_final)

afile = open(savedir+prefix+'scheds.pkl', 'wb')
pickle.dump(schedules, afile)
afile.close()

afile = open(savedir+prefix+'works.pkl', 'wb')
pickle.dump(all_works, afile)
afile.close()

