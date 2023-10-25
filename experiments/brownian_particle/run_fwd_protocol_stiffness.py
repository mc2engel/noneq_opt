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
batch_size= int(sys.argv[2])#1e5
temperature = 1. #kBT
key = random.PRNGKey(0)
key, split = random.split(key)

init_position = 15.*jnp.ones((N,dim)) #particle initial position
k_init = float(sys.argv[3])#0.2
k_final = float(sys.argv[4])#1.
trap_r0 = 15.
teq=10.
Neq=int(teq/dt)

#schedule_ = jnp.array([0.2891954,  0.2909021,  0.2950013,  0.29923955, 0.30025023, 0.3047697, 0.3077084,  0.3115996 ]) 
schedule_ = jnp.linspace(k_init, k_final, 8)


energy_fn = bp.Vspring_stiff(trap_r0)
force_fn = quantity.force(energy_fn)
displacement_fn, shift_fn = space.free()

init_schedule = jnp.linspace(k_init, k_final, 8)


mapped_sim_theory = vmap(bp.run_brownian_theoretical_opt_stiffness, [None, None, None, None, None, 0, None, None, None, None, None])
mapped_sim_ = vmap(bp.run_brownian_opt_stiffness, [None, None, None, None, None, None, 0, None, None, None, None])

seeds = jax.random.split(split, batch_size)

_, _, works_th = mapped_sim_theory(energy_fn, k_init, k_final, Neq, shift_fn, seeds, simulation_steps, end_time, dt, temperature, init_position)
positions, log_probs, works = mapped_sim_(energy_fn, schedule_, k_init, k_final, Neq, shift_fn, seeds, simulation_steps, dt, temperature, init_position)
print(jnp.array(works_th).shape)
print("AD average work: ",jnp.mean(jnp.sum(works,axis=1)), "theoretical avg work: ", jnp.mean(jnp.sum(works_th,axis=1)))
#print("theoretical avg work: ", jnp.mean(jnp.sum(works_th, axis=0)))
#positions = jnp.reshape(positions,positions.shape[1])
#works = jnp.reshape(works,works.shape[1])
print(positions.shape)
last_pos = positions[:,-1,1,1].reshape(positions.shape[0])
print(onp.mean((last_pos-trap_r0)*(last_pos-trap_r0)))
file_ = open('output.txt','w')
#works = jnp.concatenate([jnp.reshape(0., [1]),works])
#onp.savetxt(file_, onp.stack((positions,works),axis=-1))
onp.savetxt(file_,last_pos)
file_.close()


