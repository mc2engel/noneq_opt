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
from matplotlib import rc
import seaborn as sns
import pandas as pd
import pickle
import csv
import sys
import tqdm
import time

from noneq_opt import brownian_particle as bp

CB91_Blue = '#2CBDFE'
CB91_Green = '#47DBCD'
CB91_Pink = '#F3A0F2'
CB91_Purple = '#9D2EC5'
CB91_Violet = '#661D98'
CB91_Amber = '#F5B14C'

color_list = [CB91_Blue, CB91_Pink, CB91_Green, CB91_Amber,
              CB91_Purple, CB91_Violet]
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=color_list)

####Load data###

end_time = float(sys.argv[1])
opt_steps = int(sys.argv[2])
batch_size = int(sys.argv[3])
k_init = float(sys.argv[4])
k_final = float(sys.argv[5])

loaddir = '/n/brenner_lab/User/mengel/PROJECTS/NONEQ_REFACTOR_2023/noneq_opt/experiments/brownian_particle/output/'
prefix = 'stiffness_particle_simtime_%.3f_optsteps_%i_batch_%i_kinit_%.3f_kfinal_%.3f'%(end_time,opt_steps,batch_size,k_init,k_final)

file2 = open(loaddir+prefix+'scheds.pkl', 'rb')
schedules = pickle.load(file2)
file2.close()

file2 = open(loaddir+prefix+'works.pkl', 'rb')
all_works = pickle.load(file2)
file2.close()


####COMPUTE THEORETICAL MIN WORK####
c2t = (-1.-k_final*end_time+jnp.sqrt(1.+2.*k_init*end_time+k_final*k_init*end_time**2))/(2.+k_final*end_time)

work_min_th = (c2t)**2/(k_init*end_time) - jnp.log(1.+c2t) + 0.5*(k_final/k_init)*(1.+c2t)**2-0.5

print(work_min_th)

##### PLOT LOSS AND SCHEDULE EVOLUTION #####
labelfont=26
tickfont=26
lw_ = 3
r0_init = 15.
dt = 1e-3
simulation_steps = int((end_time)/dt)+1 
init_schedule = jnp.linspace(k_init, k_final, 8)
trap_fn = bp.make_trap_fxn(init_schedule, simulation_steps, k_init,k_final)
init_sched = trap_fn(jnp.arange(simulation_steps))

_, (ax0, ax1) = plt.subplots(1, 2, figsize=[30, 12])
bp.plot_with_stddev(all_works.T, ax=ax0)
ax0.hlines(y=work_min_th,xmin=0,xmax=opt_steps,linewidth=lw_,linestyle='--',color='r')
ax0.set_xlabel("Optimization iteration",fontsize=labelfont)
ax0.set_ylabel("Dissipated work (kT)",fontsize=labelfont)
ax0.tick_params(axis='x', labelsize=tickfont)
ax0.tick_params(axis='y', labelsize=tickfont)
colors = ['']
#ax0.set_ylim(0.46,0.5)
ax1.set_ylabel("Trap stiffness",fontsize=labelfont)
ax1.set_xlabel("Time",fontsize=labelfont)
ax1.tick_params(axis='x', labelsize=tickfont)
ax1.tick_params(axis='y', labelsize=tickfont)
ax1.plot(jnp.arange(simulation_steps)*dt, init_sched,lw=lw_, label='Initial Guess')

index=0
for j, sched in schedules:
  if((j == 30) or (j==60) or (j==100) or (j==400)):
    index+=1
    trap_fn = bp.make_trap_fxn(sched,simulation_steps,k_init,k_final)
    full_sched = trap_fn(jnp.arange(simulation_steps))
    ax1.plot(jnp.arange(simulation_steps)*dt, full_sched,lw=lw_, label=f'Step {j}')

#plot final:
j, sched = schedules[-1]
trap_fn = bp.make_trap_fxn(sched,simulation_steps,k_init,k_final)
full_sched = trap_fn(jnp.arange(simulation_steps))
ax1.plot(jnp.arange(simulation_steps)*dt, full_sched, lw=lw_, label="Final")

#theoretical prediction:
lambda_t = bp.make_trap_fxn_theoretical_lambda_k(dt, simulation_steps, end_time, k_init, k_final)
trap=[]
for k in (jnp.arange(simulation_steps)):
  trap.append(lambda_t(k))
ax1.plot(jnp.arange(simulation_steps)*dt, trap, '--',lw=3, label='Theory')

plt.legend(fontsize=labelfont)
plt.savefig('visualize_stiff_protocol_lambda_i_%.3f_lambda_f_%.3f_lessequiltime.png'%(k_init,k_final),format='png',dpi=400)
#plt.show()

