
from noneq_opt import barrier_crossing

import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.animation import ArtistAnimation
from matplotlib import rc
from matplotlib import colors
import seaborn as sns
rc('animation', html='jshtml')
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import sys
import jax.numpy as jnp
import pickle
from jax import random
import time
import pdb 

####load data#####

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

#landscape params:
x_m=10. #nm #25/2 == rough contour length of unfolded 44 bp hairpin
delta_E=0 #
#kappa_l=21.3863/(beta*x_m**2)
kappa_l = kappa/(beta*x_m*2)
kappa_r=kappa_l #pN/nm 
k_trap = 0.4 # pN/nm 
init_position=r0_init*jnp.ones((N,dim))

#load files:
loaddir = 'output/'
#prefix = 'end_time_%.10f_dt_%.10f_chebdeg_%i_kappa_%.4f_batch_%i_trainsteps_%i_' %(end_time, dt,degree,kappa,batch_size,opt_steps)
prefix = 'end_time_%i_dt_%i_chebdeg_%i_kappa_%.3f_batch_%i_trainsteps_%i_' %(end_time, dt,degree,kappa,batch_size,opt_steps)


afile = open(loaddir+prefix+'all_works_REINFORCE.pkl', 'rb')
all_works_REINFORCE = pickle.load(afile)
afile.close()

afile = open(loaddir+prefix+'coeffs_REINFORCE.pkl', 'rb')
coeffs_REINFORCE = pickle.load(afile)
afile.close()

afile = open(loaddir+prefix+'grads_REINFORCE.pkl', 'rb')
grads_REINFORCE = jnp.array(pickle.load(afile))
afile.close()

afile = open(loaddir+prefix+'grads_REINFORCE_logP_vals.pkl', 'rb')
grads_REINFORCE_logP_vals = jnp.array(pickle.load(afile))
afile.close()

afile = open(loaddir+prefix+'grads_REINFORCE_reparam_vals.pkl', 'rb')
grads_REINFORCE_reparam_vals = jnp.array(pickle.load(afile))
afile.close()

afile = open(loaddir+prefix+'all_works_logP.pkl', 'rb')
all_works_logP = pickle.load(afile)
afile.close()

afile = open(loaddir+prefix+'coeffs_logP.pkl', 'rb')
coeffs_logP = pickle.load(afile)
afile.close()

afile = open(loaddir+prefix+'grads_logP.pkl', 'rb')
grads_logP = jnp.array(pickle.load(afile))
afile.close()

afile = open(loaddir+prefix+'all_works_reparam.pkl', 'rb')
all_works_reparam = pickle.load(afile)
afile.close()

afile = open(loaddir+prefix+'coeffs_reparam.pkl', 'rb')
coeffs_reparam = pickle.load(afile)
afile.close()

afile = open(loaddir+prefix+'grads_reparam.pkl', 'rb')
grads_reparam = jnp.array(pickle.load(afile))
afile.close()

fontsize=20
##### PLOT LOSS#####
#_, (ax0) = plt.subplots(1, 1, figsize=[12, 12])
plt.figure()
barrier_crossing.plot_with_stddev(jnp.array(all_works_REINFORCE).T, label='REINFORCE', ax=plt)
barrier_crossing.plot_with_stddev(jnp.array(all_works_logP).T, label='logP', ax=plt)
barrier_crossing.plot_with_stddev(jnp.array(all_works_reparam).T, label='reparam', ax=plt)
#plt.title('Total work')
plt.ylabel("Work (kT)",fontsize=fontsize)
plt.xlabel("Optimization iteration",fontsize=fontsize)
plt.legend()
plt.tick_params(axis='both', which='major', labelsize=18)
plt.show()

#####PLOT PROTOCOL EVOLUTION#####
plt.figure()
init_coeffs = coeffs_REINFORCE[0][1]
trap_fn = barrier_crossing.make_trap_fxn(jnp.arange(simulation_steps+1),init_coeffs,r0_init,r0_final)
init_sched = trap_fn(jnp.arange(simulation_steps+1))
plt.plot(jnp.arange(simulation_steps+1), init_sched, label='initial guess')

#for j, coeffs in coeffs_:
#  if(j%1 == 0):
#    trap_fn = make_trap_fxn(jnp.arange(simulation_steps+1),coeffs,r0_init,r0_final)
#    full_sched = trap_fn(jnp.arange(simulation_steps+1))
#    ax1.plot(jnp.arange(simulation_steps+1), full_sched, '-', label=f'Step {j}')

#plot final estimate:

trap_fn_REINFORCE = barrier_crossing.make_trap_fxn(jnp.arange(simulation_steps+1),coeffs_REINFORCE[-1][1],r0_init,r0_final)
full_sched_REINFORCE = trap_fn_REINFORCE(jnp.arange(simulation_steps+1))
plt.plot(jnp.arange(simulation_steps+1), full_sched_REINFORCE, '-', label=f'Final REINFORCE')

trap_fn_logP = barrier_crossing.make_trap_fxn(jnp.arange(simulation_steps+1),coeffs_logP[-1][1],r0_init,r0_final)
full_sched_logP = trap_fn_logP(jnp.arange(simulation_steps+1))
plt.plot(jnp.arange(simulation_steps+1), full_sched_logP, '-', label=f'Final logP')

trap_fn_reparam = barrier_crossing.make_trap_fxn(jnp.arange(simulation_steps+1),coeffs_reparam[-1][1],r0_init,r0_final)
full_sched_reparam = trap_fn_reparam(jnp.arange(simulation_steps+1))
plt.plot(jnp.arange(simulation_steps+1), full_sched_reparam, '-', label=f'Final reparam')

plt.legend()#
#plt.title('Schedule evolution')
plt.xlabel("Simulation time",fontsize=fontsize)
plt.ylabel("Trap position lambda(t) (nm)",fontsize=fontsize)
plt.legend()
plt.tick_params(axis='both', which='major', labelsize=18)
plt.show()

####PLOT GRADIENTS#####
a = 0
b = 1
c = 2
fig, (ax0, ax1, ax2) = plt.subplots(1, 3, sharex=True, figsize=[24,8])
ax0.plot(jnp.arange(opt_steps), grads_REINFORCE[:,a], label="REINFORCE")
ax0.plot(jnp.arange(opt_steps), grads_logP[:,a], label="logP")
ax0.plot(jnp.arange(opt_steps), grads_reparam[:,a], label="reparam")

ax1.plot(jnp.arange(opt_steps), grads_REINFORCE[:,b], label="REINFORCE")
ax1.plot(jnp.arange(opt_steps), grads_logP[:,b], label="logP")
ax1.plot(jnp.arange(opt_steps), grads_reparam[:,b], label="reparam")

ax2.plot(jnp.arange(opt_steps), grads_REINFORCE[:,c], label="REINFORCE")
ax2.plot(jnp.arange(opt_steps), grads_logP[:,c], label="logP")
ax2.plot(jnp.arange(opt_steps), grads_reparam[:,c], label="reparam")
#plt.title("Gradients")
fig.add_subplot(111, frameon=False)
plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
plt.ylabel("Gradient",fontsize=fontsize)
plt.xlabel("Optimization iteration",fontsize=fontsize)
ax0.set_title("Coefficient %i"%a)
ax1.set_title("Coefficient %i"%b)
ax2.set_title("Coefficient %i"%c)

ax0.legend()
ax1.legend()
ax2.legend()
plt.show()

####PLOT GRADIENTS, following the REINFORCE optimization curve, but lookign at teh size of each term at each step#####
pdb.set_trace()
fig, (ax0, ax1, ax2) = plt.subplots(1, 3, sharex=True, figsize=[24,8])
ax0.plot(jnp.arange(opt_steps), grads_REINFORCE[:,a], label="REINFORCE")
ax0.plot(jnp.arange(opt_steps), grads_REINFORCE_logP_vals[:,a], label="logP")
ax0.plot(jnp.arange(opt_steps), grads_REINFORCE_reparam_vals[:,a], label="reparam")

ax1.plot(jnp.arange(opt_steps), grads_REINFORCE[:,b], label="REINFORCE")
ax1.plot(jnp.arange(opt_steps), grads_REINFORCE_logP_vals[:,b], label="logP")
ax1.plot(jnp.arange(opt_steps), grads_REINFORCE_reparam_vals[:,b], label="reparam")

ax2.plot(jnp.arange(opt_steps), grads_REINFORCE[:,c], label="REINFORCE")
ax2.plot(jnp.arange(opt_steps), grads_REINFORCE_logP_vals[:,c], label="logP")
ax2.plot(jnp.arange(opt_steps), grads_REINFORCE_reparam_vals[:,c], label="reparam")
#plt.title("Gradients")
fig.add_subplot(111, frameon=False)
plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
plt.ylabel("Gradient",fontsize=fontsize)
plt.xlabel("Optimization iteration",fontsize=fontsize)
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
