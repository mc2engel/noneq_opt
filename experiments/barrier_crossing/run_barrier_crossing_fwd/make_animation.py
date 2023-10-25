#mport jax.tools.colab_tpu
#jax.tools.colab_tpu.setup_tpu()

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

#from jax.experimental import stax
from jax.example_libraries import optimizers as jopt

#from jax.config import config
#config.update('jax_enable_x64', False)

from jax_md import space
from jax_md import minimize
from jax_md import simulate
from jax_md import space
from jax_md import energy
from jax_md import quantity
from jax_md import smap
import jax_md as jmd

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
import seaborn as sns
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.ticker as ticker


import tqdm
import time
import typing

import scipy as sp
import scipy.special as sps

from numpy import polynomial
import numpy as onp 

#import jax.profiler
#server = jax.profiler.start_server(port=1234)

from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions

from noneq_opt import barrier_crossing


#parameters to plot:
end_time = int(sys.argv[1]) #128
dt = int(sys.argv[2]) #11 
#degree of Chebyshev polynomials that describe protocols
degree = int(sys.argv[3]) #32
kappa = int(sys.argv[4])
batch_size = int(sys.argv[5]) #256
training_steps = int(sys.argv[6]) #10 
plot_theory = int(sys.argv[7])
savestring = str(sys.argv[8])

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

#landscape params:
x_m=14. #nm #25/2 == rough contour length of unfolded 44 bp hairpin
delta_E=0 #
#kappa_l=21.3863/(beta*x_m**2)
kappa_l = kappa
kappa_r=kappa_l #pN/nm 
k_trap = 0.4 # pN/nm 
init_position=r0_init*jnp.ones((N,dim))
key = random.PRNGKey(int(time.time()))
key, split1 = random.split(key, 2)

displacement_fn, shift_fn = space.free()


####load protocol files:####


loaddir = '../run_barrier_crossing_opt/output/'
prefix = 'end_time_%i_dt_%i_chebdeg_%i_kappa_%.3f_batch_%i_trainsteps_%i_' %(end_time, dt,degree,kappa_l,batch_size,training_steps)

if(plot_theory==0):
    file1 = open(loaddir+prefix+'coeffs.pkl', 'rb')
    coeffs = pickle.load(file1)
    file1.close()
    
    file2 = open(loaddir+prefix+'all_works.pkl', 'rb')
    all_works = pickle.load(file2)
    file2.close()


if(plot_theory == 1):
    ###load theoretical protocols###
    theoretical_schedule_2kT = onp.load("../theoretical_protocol_sivak_crooks_2.5kT.npy")
    tvec_2kT=theoretical_schedule_2kT[:,0]
    lambda_t_2kT =theoretical_schedule_2kT[:,1]
    
    #make trap fxns:
    theory_trap_fn_2kT = make_theoretical_trap_fxn(jnp.arange(simulation_steps), dt, tvec_2kT, lambda_t_2kT)
    positions_2kT = theory_trap_fn_2kT(jnp.arange(simulation_steps))

if(plot_theory == 2):
    theoretical_schedule_10kT = onp.load("../theoretical_protocol_sivak_crooks_10kT.npy")
    tvec_10kT=theoretical_schedule_10kT[:,0]
    lambda_t_10kT =theoretical_schedule_10kT[:,1]
    
    #make trap fxns:
    theory_trap_fn_10kT = make_theoretical_trap_fxn(jnp.arange(simulation_steps), dt, tvec_10kT, lambda_t_10kT)
    positions_10kT = theory_trap_fn_10kT(jnp.arange(simulation_steps))


####load forward simulation files:####

loaddir = 'output/'
prefix = 'end_time_%i_dt_%i_kappa_%.3f_batch_%i' %(end_time, dt,kappa_l,batch_size)

afile = open(loaddir+prefix+savestring+'all_pos.pkl', 'rb')
all_pos = pickle.load(afile)
afile.close()

afile = open(loaddir+prefix+savestring+'all_works.pkl', 'rb')
all_works = pickle.load(afile)
afile.close()


#plot params
labelsize=26
ticksize=16
legendsize=18
plt.rcParams['axes.linewidth'] = 1 
insetlabels=16
lincolor = 'tab:grey'
ADcolor= 'tab:orange'
thcolor= 'tab:cyan'

num_curves = 6
baseline_palette = sns.color_palette('YlGnBu', n_colors=num_curves+1)[1:]
theory_colour = sns.xkcd_palette(['burnt orange'] * 2)
sns.palplot(baseline_palette + theory_colour[:1])

palette_ = baseline_palette + theory_colour[:1]


#load potential function:

x=jnp.linspace(-20,20,200)
kappa_l_10kT=21.3863/(beta*x_m**2) #pN/nm #for Ebarrier = 10kT and delta_E=0, as C&S use
kappa_l_2kT=6.38629/(beta*x_m**2) #pN/nm #for Ebarrier = 2.5kT and delta_E=0, as C&S use


xvec=jnp.reshape(jnp.linspace(-20,20,200), [200,1,1])
k_splot = 0.
Vfn = V_biomolecule(kappa_l, kappa_l, x_m, delta_E, k_splot, beta) #returns in pN nm
Vpotential = []
for j in range(len(xvec)):
  Vpotential.append(Vfn(xvec[j], r0=0.)*beta)

trap_fn = barrier_crossing.make_trap_fxn(jnp.arange(simulation_steps), coeffs[-1][-1], r0_init, r0_final)

###compute running average of the mean work

#average the works:
av_works = jnp.mean(works_10kT_AD,axis=0)*beta

from scipy.ndimage.filters import uniform_filter1d
N = 1000
y = uniform_filter1d(av_works, size=N)

def _avg(Wtot,input_):
  n, Wn = input_
  Wtot+=Wn
  return Wtot, (Wtot)

numits=jnp.arange(1, len(jnp.array(y))+1)
input_ = numits, jnp.array(y)
tot_work, cum_avg = jax.lax.scan(_avg, 0.0, input_)

#####MAKE ANIMATION#####
import functools
#from noneq_opt import parameterization as p10n
import noneq_opt.barrier_crossing as xing
from jax_md import space, energy

#additional params for plotting animation:
plot_every = 50
max_energy = 15
mol_location = -10.
times = onp.linspace(0, end_time, simulation_steps // plot_every)
steps = jnp.arange(0, simulation_steps)
works = cum_avg[0:-1:plot_every]
#landscape params:
x_m=10. #nm
delta_E=0 #pN nm
#kappa_l=21.3863/(beta*x_m**2) #pN/nm #for Ebarrier = 10kT and delta_E=0, as C&S use
kappa_l=6.38629/(beta*x_m**2) #pN/nm #for Ebarrier = 2.5kT and delta_E=0, as C&S use
kappa_r=kappa_l #pN/nm #I think this is what Crooks and Sivak do...


displacement_fn, _ = space.free()

xs = jnp.linspace(-2 * mol_location, 2 * mol_location, 100)[..., jnp.newaxis]

def trap_potential_fn(central_pos_fn, k):
  def _potential(position, t):
    d = position - central_pos_fn(t)
    return energy.simple_spring(d, epsilon=k, length=0).sum()
  return _potential


#energy_fn_2kT = V_biomolecule(kappa_l_2kT, kappa_r_2kT, x_m, delta_E, k_s, beta)
#energy_fn_10kT = V_biomolecule(kappa_l_10kT, kappa_r_10kT, x_m, delta_E, k_s, beta)
#trap_fn_2kT_AD = make_trap_fxn(jnp.arange(simulation_steps), coeffs_2kT_load[-1][-1], r0_init, r0_final)
#trap_fn_10kT_AD = make_trap_fxn(jnp.arange(simulation_steps), coeffs_10kT_load[-1][-1], r0_init, r0_final)
#trap_fn_2kT_theory = make_theoretical_trap_fxn(jnp.arange(simulation_steps), dt, tvec_2kT, lambda_t_2kT)
#trap_fn_10kT_theory = make_theoretical_trap_fxn(jnp.arange(simulation_steps), dt, tvec_10kT, lambda_t_10kT)
#trap_fn_linear = make_trap_fxn(jnp.arange(simulation_steps), init_coeffs_linear, r0_init, r0_final)

trap_fn = functools.partial(trap_potential_fn, k=k_s)#, trap_fn_2kT_AD, k_s)
trp = jax.vmap(trap_fn(trap_fn_2kT_theory), [0, None])
xs = jnp.linspace(-2 * mol_location, 2 * mol_location, 100)[..., jnp.newaxis]

def V_molecule(kappa_l, kappa_r, x_m, delta_E, beta):
  def total_energy(position, r0=0.0, **unused_kwargs):
      x = position
      #underlying energy landscape:
      Em = -(1./beta)*jnp.log(jnp.exp(-0.5*beta*kappa_l*(x+x_m)**2)+jnp.exp(-(0.5*beta*kappa_r*(x-x_m)**2+beta*delta_E)))
      return Em
  return total_energy
#molecule = V_biomolecule_(kappa_l, kappa_r, x_m, delta_E, 0.0, beta)
molecule = V_molecule(kappa_l, kappa_r, x_m, delta_E, beta)

fig, ax = plt.subplots(1, 2, figsize=[13, 6])
plt.rc('xtick', labelsize=20) 
plt.rc('ytick', labelsize=20) 
#fig.tight_layout(pad=1.5)
plt.subplots_adjust(left=0.2, bottom=None, right=None, top=None, wspace=0.24, hspace=None)
def frame(j):
  runtime = times[:j + 1]
  qax = ax[1]
  qax.clear();
  quantity = works[:j+1]
  qax.plot(1000*runtime, quantity, 'c-',label='Linear')
  qax.plot(1000*times[j], quantity[-1], 'co')
  qax.set_xlim(1000*times[0], 1000*times[-1])
  qax.set_ylim(-0.0001, 2.5)
  qax.set_xlabel("Time (ms)",fontsize=20)
  qax.set_ylabel("Total Work (kT)",fontsize=20)
  qax.legend()

  pax = ax[0]
  pax.clear()
  trp_ = trp(xs, j*plot_every)
  mol = jax.vmap(molecule, [0])(xs)
  nrg = molecule(all_positions[:,j*plot_every,0,0]) + trp(all_positions[:,j*plot_every,0,0],steps[j*plot_every])
  total_nrg = trp(xs,steps[j*plot_every])+mol[:,0]
  pax.plot(xs, beta*trp(xs,steps[j*plot_every]), 'r-', lw = 3, label='Trap')
  pax.plot(xs, mol*beta, 'b-', lw = 3, label='Molecular Potential')
  pax.plot(xs, total_nrg*beta, 'm-', lw=3, label='Total')
  pax.scatter(all_positions[:,j*plot_every,0,0], (nrg+1)*beta, c='g', marker='.', s=100, label='locations')
  pax.hist
  pax.set_ylim(-2, max_energy)
  pax.set_xlim(xs[-1, 0], xs[0, 0])
  #ax.text(2 * mol_location, max_energy+2, f'Time: {times[j*plot_every]:.2f}',fontsize=20)
  pax.legend(loc=1)
  pax.set_xlabel("Position (nm)",fontsize=20)
  pax.set_ylabel("Energy (kT)",fontsize=20)
  #labsx = pax.get_xticklabels()
  #pax.set_xticklabels(labsx,fontsize=20)
  #pax.set_yticklabels([],fontsize=20)

##second animated subplot: contains average work vs time for linear, AD, theory protocols and a marker that traces the curve 
#as the video of particles surmounting the barrier plays
##???

#frame(0)
#print('Building animation...')
#anim = animation.FuncAnimation(
#    fig, frame, blit=True, frames=(simulation_steps // plot_every + 1))
#plt.close(fig)
#anim

ani = animation.FuncAnimation(fig, frame, frames=(simulation_steps // plot_every))
plt.close(fig)
ani

#save
file_ = r"/content/drive/MyDrive/Colab Notebooks/Barrier_crossing/movies/animation_2kT_linear.mp4"
writervideo = animation.FFMpegWriter(fps=40) 
ani.save(file_, writer=writervideo, dpi=300)

