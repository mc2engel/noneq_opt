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
import pdb
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

savedir = '/n/brenner_lab/User/mengel/PROJECTS/NONEQ_REFACTOR_2023/noneq_opt/experiments/brownian_particle/output/'
prefix = 'drag_particle_simtime_%.3f_optsteps_%i_batch_%i_'%(end_time,opt_steps,batch_size)

afile = open(savedir+prefix+'scheds_REINFORCE.pkl', 'rb')
schedules_REINFORCE = pickle.load(afile)
afile.close()

afile = open(savedir+prefix+'works_REINFORCE.pkl', 'rb')
all_works_REINFORCE = pickle.load(afile)
afile.close()

afile = open(savedir+prefix+'grads_REINFORCE.pkl', 'rb')
grads_REINFORCE = pickle.load(afile)
afile.close()

afile = open(savedir+prefix+'grads_logP_vals.pkl', 'rb')
grads_logP_vals = pickle.load(afile)
afile.close()

afile = open(savedir+prefix+'grads_reparam_vals.pkl', 'rb')
grads_reparam_vals = pickle.load(afile)
afile.close()

afile = open(savedir+prefix+'scheds_logP.pkl', 'rb')
schedules_logP = pickle.load(afile)
afile.close()

afile = open(savedir+prefix+'works_logP.pkl', 'rb')
all_works_logP = pickle.load(afile)
afile.close()

afile = open(savedir+prefix+'grads_logP.pkl', 'rb')
grads_logP = pickle.load(afile)
afile.close()

afile = open(savedir+prefix+'scheds_reparam.pkl', 'rb')
schedules_reparam = pickle.load(afile)
afile.close()

afile = open(savedir+prefix+'works_reparam.pkl', 'rb')
all_works_reparam = pickle.load(afile)
afile.close()

afile = open(savedir+prefix+'grads_reparam.pkl', 'rb')
grads_REINFORCE = pickle.load(afile)
afile.close()

###COMPUTE THEORETICAL MIN WORK####

work_th_min = 5.*5./(end_time+2)

##### PLOT LOSS #####
labelfont=26
tickfont=26
lw_ = 5
r0_init = 5.
r0_final = 10.
dt = 1e-3
simulation_steps = int((end_time)/dt)+1 
init_schedule = jnp.linspace(r0_init, r0_final, 8)
trap_fn = bp.make_trap_fxn(init_schedule, simulation_steps, r0_init,r0_final)
init_sched = trap_fn(jnp.arange(simulation_steps))

_, (ax0, ax1) = plt.subplots(1, 2, figsize=[30, 12])
bp.plot_with_stddev(all_works_REINFORCE.T, ax=ax0, label='REINFORCE',color='r')
bp.plot_with_stddev(all_works_logP.T, ax=ax0, label='logP',color='b')
bp.plot_with_stddev(all_works_reparam.T, ax=ax0, label='reparam',color='k')
ax0.hlines(y=work_th_min,xmin=0,xmax=opt_steps,linewidth=lw_,linestyle='--',color='r')
ax0.set_xlabel("Optimization iteration",fontsize=labelfont)
ax0.set_ylabel("Dissipated work (kT)",fontsize=labelfont)
ax0.tick_params(axis='x', labelsize=tickfont)
ax0.tick_params(axis='y', labelsize=tickfont)
colors = ['']
ax0.legend()
#ax0.set_ylim(0.,4.)
#ax0.set_ylim(10.5,12.)

ax1.set_ylabel("Trap position",fontsize=labelfont)
ax1.set_xlabel("Time",fontsize=labelfont)
ax1.tick_params(axis='x', labelsize=tickfont)
ax1.tick_params(axis='y', labelsize=tickfont)
ax1.plot(jnp.arange(simulation_steps)*dt, init_sched,lw=lw_, label='Initial Guess')

#plot final REINFORCE:
j, scheds_REINFORCE = schedules_REINFORCE[-1]
trap_fn = bp.make_trap_fxn(scheds_REINFORCE,simulation_steps,r0_init,r0_final)
full_sched = trap_fn(jnp.arange(simulation_steps))
ax1.plot(jnp.arange(simulation_steps)*dt, full_sched, lw=lw_, label="REINFORCE")
#plot final reparam:
j, scheds_logP = schedules_logP[-1]
trap_fn = bp.make_trap_fxn(scheds_logP,simulation_steps,r0_init,r0_final)
full_sched = trap_fn(jnp.arange(simulation_steps))
ax1.plot(jnp.arange(simulation_steps)*dt, full_sched, lw=lw_, label="logP")
#plot final logP:
j, scheds_reparam = schedules_reparam[-1]
trap_fn = bp.make_trap_fxn(scheds_reparam,simulation_steps,r0_init,r0_final)
full_sched = trap_fn(jnp.arange(simulation_steps))
ax1.plot(jnp.arange(simulation_steps)*dt, full_sched, lw=lw_, label="reparam")

plt.legend(fontsize=labelfont)
#plt.savefig('visualize_r0_protocol_%.3f.png'%(end_time),format='png',dpi=400)
plt.show()


####PLOT GRADIENTS#####
a = 0
b = 3
c = 7
pdb.set_trace()
fig, (ax0, ax1, ax2) = plt.subplots(1, 3, sharex=True, figsize=[24,8])
ax0.plot(jnp.arange(opt_steps), jnp.array(grads_REINFORCE)[:,a], label="REINFORCE")
ax0.plot(jnp.arange(opt_steps), jnp.array(grads_logP_vals)[:,a], label="logP")
ax0.plot(jnp.arange(opt_steps), jnp.array(grads_reparam_vals)[:,a], label="reparam")
ax0.set_ylabel("Gradient",fontsize=labelfont)
ax0.set_xlabel("Optimization iteration",fontsize=labelfont)
ax1.plot(jnp.arange(opt_steps), jnp.array(grads_REINFORCE)[:,b], label="REINFORCE")
ax1.plot(jnp.arange(opt_steps), jnp.array(grads_logP_vals)[:,b], label="logP")
ax1.plot(jnp.arange(opt_steps), jnp.array(grads_reparam_vals)[:,b], label="reparam")
ax1.set_xlabel("Optimization iteration",fontsize=labelfont)

ax2.plot(jnp.arange(opt_steps), jnp.array(grads_REINFORCE)[:,c], label="REINFORCE")
ax2.plot(jnp.arange(opt_steps), jnp.array(grads_logP_vals)[:,c], label="logP")
ax2.plot(jnp.arange(opt_steps), jnp.array(grads_reparam_vals)[:,c], label="reparam")
ax2.set_xlabel("Optimization iteration",fontsize=labelfont)
#plt.title("Gradients")
#fig.add_subplot(111, frameon=False)
#plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
#fig.tight_layout()
#plt.ylabel("Gradient",fontsize=labelfont)
plt.xlabel("Optimization iteration",fontsize=labelfont)
ax0.set_title("Coefficient %i"%a)
ax1.set_title("Coefficient %i"%b)
ax2.set_title("Coefficient %i"%c)
ax1.tick_params(axis='both', which='major', labelsize=18)
ax0.tick_params(axis='both', which='major', labelsize=18)
ax2.tick_params(axis='both', which='major', labelsize=18)
ax0.legend()
ax1.legend()
ax2.legend()
plt.show()


