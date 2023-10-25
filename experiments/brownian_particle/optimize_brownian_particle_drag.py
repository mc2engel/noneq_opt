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
box_size = 20
end_time = float(sys.argv[1])#2.69
dt = 1e-3 #this needs to be sufficiently small.. check for strange behaviour!
simulation_steps = int((end_time)/dt)+1 #{0, dt, 2dt, ..., end time} add 1 for t=0
print("running simulation for ",simulation_steps," steps.")
opt_steps = int(sys.argv[2]) #100
batch_size= int(sys.argv[3]) #5000
temperature = 1. #kBT
key = random.PRNGKey(0)
key, split = random.split(key)
#mass is default = 1
#gamma is set to 1.
#so diffusion constant D is D = 1./(beta*gamma*mass) #s^(-1) = kT = 1.0 here

init_position = 5. #particle initial position
r0_init = 5.
r0_final = 10.
trap_stiffness = 1.
Neq=100 # should be fine for starting at r0!

optimizer = jopt.adam(.1)

energy_fn = bp.Vspring_r0(box_size, trap_stiffness)
force_fn = quantity.force(energy_fn)
displacement_fn, shift_fn = space.periodic(box_size)

init_schedule = jnp.linspace(5., 10., 8)
trap_fn = bp.make_trap_fxn(init_schedule, simulation_steps, r0_init, r0_final)

summaries = []
schedules = []
all_works = []
opt_state = optimizer.init_fn(init_schedule)

grad_fxn = bp.estimate_gradient_r0(batch_size, energy_fn, r0_init, r0_final, Neq, shift_fn, simulation_steps, dt, temperature)

for j in tqdm.trange(opt_steps,position=0):
  key, split = random.split(key)
  grad, (works, summary) = grad_fxn(optimizer.params_fn(opt_state), split)
  opt_state = optimizer.update_fn(j, grad, opt_state)
  all_works.append(works)
  #summaries.append(summary)
  #if j % (opt_steps // 10) == 0:
  schedules.append((j,) + (optimizer.params_fn(opt_state),))

print("final parameters: ", optimizer.params_fn(opt_state))
#all_summaries = jax.tree_multimap(lambda *args: jnp.stack(args), *summaries)
all_works = jax.tree_map(lambda *args: jnp.stack(args), *all_works)


savedir = '/n/brenner_lab/User/mengel/PROJECTS/NONEQ_REFACTOR_2023/noneq_opt/experiments/brownian_particle/output/'
prefix = 'drag_particle_simtime_%.3f_optsteps_%i_batch_%i_'%(end_time,opt_steps,batch_size)

afile = open(savedir+prefix+'scheds.pkl', 'wb')
pickle.dump(schedules, afile)
afile.close()

afile = open(savedir+prefix+'works.pkl', 'wb')
pickle.dump(all_works, afile)
afile.close()

