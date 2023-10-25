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
box_size = 20.

init_position = 5.*jnp.ones((N,dim)) #particle initial position
#init_position = 5.
r0_init = float(sys.argv[3])#0.2
r0_final = float(sys.argv[4])#1.
trap_stiffness = 1.
teq=1.
Neq=int(teq/dt)

schedule_ = jnp.array([0.2891954,  0.2909021,  0.2950013,  0.29923955, 0.30025023, 0.3047697, 0.3077084,  0.3115996 ]) 
#init_schedule = jnp.linspace(k_init, k_final, 8)


energy_fn = bp.Vspring_r0(box_size, trap_stiffness)
force_fn = quantity.force(energy_fn)
displacement_fn, shift_fn = space.periodic(box_size)

init_schedule = jnp.linspace(r0_init, r0_final, 8)


mapped_sim_theory = vmap(bp.run_brownian_theoretical_opt_r0, [None, None, None, None, None, 0, None, None, None, None])
mapped_sim_ = vmap(bp.run_brownian_opt_r0, [None, None, None, None, None, None, 0, None, None, None])

seeds = jax.random.split(split, batch_size)

_, _, works_th = mapped_sim_theory(energy_fn, r0_init, r0_final, Neq, shift_fn, seeds, simulation_steps, end_time, dt, temperature)
_, works = mapped_sim_(energy_fn, schedule_, r0_init, r0_final, Neq, shift_fn, seeds, simulation_steps, dt, temperature)
print(jnp.array(works_th).shape)
print("AD average work: ",jnp.mean(jnp.sum(works,axis=1)), "theoretical avg work: ", jnp.mean(jnp.sum(works_th,axis=1)))
#print("theoretical avg work: ", jnp.mean(jnp.sum(works_th, axis=0)))

