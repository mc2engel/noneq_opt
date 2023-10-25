import jax
import jax.numpy as jnp
import jax.example_libraries.optimizers as jopt
import numpy as np

import csv
import os
import functools
import pickle

import tqdm
import time

from noneq_opt import ising
from noneq_opt import parameterization as p10n

import sys
import pdb

##simulation and training parameters

#fwd sim params:
size = int(sys.argv[1]) #32
seed = int(sys.argv[2]) #0 
time_steps = int(sys.argv[3]) #51 
samples = int(sys.argv[4]) #num forward sim batches to run

#AD opt protocol params:
prot_size = int(sys.argv[5]) #32
prot_seed = int(sys.argv[6]) #0 
prot_time_steps = int(sys.argv[7]) #51 
prot_field_degree = log_temp_degree= int(sys.argv[8]) #32
prot_batch_size = int(sys.argv[9]) #256
prot_training_steps = int(sys.argv[10])

print("Running forward Ising simulations for ",size,"x",size," lattice for ",time_steps," time steps, on ",samples," samples, using the AD-optimized protocol calculated for a ",prot_size,"x",prot_size," lattice for ",prot_time_steps," time steps, on batches of ",prot_batch_size," for ",prot_training_steps," training steps and using Chebyshev polynomials of degree ",prot_field_degree,". Near-eq'm theoretical protocol is also being run for a ",size,"x",size," lattice for ",time_steps," time steps.")

#Load the near-eqm theoretical protocol to run for comparisons:
th_temps_2015 = []
th_fields_2015 = []

thdir = '/n/brenner_lab/User/mengel/PROJECTS/NONEQ_REFACTOR_2023/noneq_opt/run_ising_opt/'
with open(thdir+'protocol_optimal_truncated.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        th_temps_2015.append(float(row[0]))
        th_fields_2015.append(float(row[1]))
      
#linearly interpolate to get a continuous schedule for the theoretical curve:
xscaledT=jnp.arange(len(th_temps_2015))/(len(th_temps_2015)-1)

th_log_temps_2015_fxn = p10n.PiecewiseLinear(jnp.log(jnp.array(th_temps_2015[1:-2])),
    y0=jnp.log(th_temps_2015[0]),
    y1=jnp.log(th_temps_2015[-1]))
th_fields_2015_fxn = p10n.PiecewiseLinear(jnp.array(th_fields_2015[1:-2]),
                                    y0 = jnp.array(th_fields_2015[0]),
                                    y1 = jnp.array(th_fields_2015[-1]))
th_schedule_2015_grant = ising.IsingSchedule(
    th_log_temps_2015_fxn, th_fields_2015_fxn)

#Load in the protocol you want to run:

loaddir = '/n/brenner_lab/User/mengel/PROJECTS/NONEQ_REFACTOR_2023/noneq_opt/launch_scripts/output/'

prefix = '%ix%ilattice_timesteps_%i_chebdeg_%i_batch_%i_trainsteps_%i_' %(prot_size,prot_size,prot_time_steps,prot_field_degree,prot_batch_size,prot_training_steps)

temp_coeffs = []
field_coeffs = []
file2 = open(loaddir+prefix+'log_temp_weights.pkl', 'rb')
temp_coeffs = pickle.load(file2)
file2.close()

file2 = open(loaddir+prefix+'field_weights.pkl', 'rb')
field_coeffs = pickle.load(file2)
file2.close()

temp_weights = temp_coeffs[-1]
field_weights = field_coeffs[-1]

#recreate schedules from loaded data:
if(prot_time_steps==11):
    Tbaseline=ising.log_temp_baseline(min_temp=0.69,degree=2.)
else:
    Tbaseline=ising.log_temp_baseline(min_temp=0.69,degree=1.)
Fbaseline =ising.field_baseline()


log_temp_schedule = p10n.AddBaseline(
    p10n.ConstrainEndpoints(
        p10n.Chebyshev(
            temp_weights
        ),
        y0=0.,
        y1=0.,
    ),
    baseline=Tbaseline
)


field_schedule = p10n.AddBaseline(
    p10n.ConstrainEndpoints(
        p10n.Chebyshev(
            field_weights
        ),
        y0=0.,
        y1=0.,
    ),
    baseline=Fbaseline
)

AD_schedule = ising.IsingSchedule(
    log_temp_schedule, field_schedule)

#set up forward sims:
times = jnp.linspace(0, 1, time_steps)
initial_spins = -jnp.ones([size, size])
parameters_th_2015 = th_schedule_2015_grant(times)
#pdb.set_trace()
parameters_AD = AD_schedule(times)


# RUN THEORY 2015:
seed_2015 = jax.random.split(jax.random.PRNGKey(int(time.time())), samples) 
mapped_sim_th_2015 = jax.vmap(lambda s: ising.simulate_ising(parameters_th_2015, initial_spins, s))
final_state_th_2015, summary_th_2015 = mapped_sim_th_2015(seed_2015)   
   
#RUN AD SIMS:
seed_AD = jax.random.split(jax.random.PRNGKey(int(time.time())), samples) 
mapped_sim_AD = jax.vmap(lambda s: ising.simulate_ising(parameters_AD, initial_spins, s))
final_state_AD, summary_AD = mapped_sim_AD(seed_AD)   


#compute entropy correction terms associated with the final ensemble probability:

#correction terms for near-eqm opt runs:
times = jnp.linspace(0, 1, time_steps)
initial_spin_state = ising.IsingState(initial_spins, ising.map_slice(parameters_th_2015, 0))
log_p_init_th = - ising.energy(initial_spin_state) / jnp.exp(parameters_th_2015.log_temp[0])
final_kT_th = jnp.exp(parameters_th_2015.log_temp[-1])

log_p_final_th = - summary_th_2015.energy[:,-1] / final_kT_th
correction_th = log_p_init_th - log_p_final_th
final_entropies_th = np.array(jax.device_get(summary_th_2015.entropy_production))
final_entropies_th[:,-1] = final_entropies_th[:,-1] + correction_th


#correction terms for AD opt runs:
times = jnp.linspace(0, 1, time_steps)
initial_spin_state = ising.IsingState(initial_spins, ising.map_slice(parameters_AD, 0))
log_p_init_AD = - ising.energy(initial_spin_state) / jnp.exp(parameters_AD.log_temp[0])
final_kT_AD = jnp.exp(parameters_AD.log_temp[-1])

log_p_final_AD = - summary_AD.energy[:,-1] / final_kT_AD
correction_AD = log_p_init_AD - log_p_final_AD
final_entropies_AD = np.array(jax.device_get(summary_AD.entropy_production))
final_entropies_AD[:,-1] = final_entropies_AD[:,-1] + correction_AD

#compute averages and std dev of the mean over the batch:

#for AD protocol:
batch_entropies_AD = np.sum(final_entropies_AD,axis=1)
avg_entropy_AD = np.sum(batch_entropies_AD)/samples
err_entropy_AD = np.std(batch_entropies_AD)/np.sqrt(samples)

#for near-eqm theory protocol:
batch_entropies_th_2015 = np.sum(final_entropies_th,axis=1)
avg_entropy_theory = np.sum(batch_entropies_th_2015)/samples
err_entropy_theory = np.std(batch_entropies_th_2015)/np.sqrt(samples)

#save results:
savedir = 'output/fwd_sims/'

prefix_AD = 'FWD_%ix%ilattice_samples_%i_USINGOPT_%ix%ilattice_chebdeg_%i_batch_%i_trainsteps_%i' %(size,size,samples,prot_size,prot_size,prot_field_degree,prot_batch_size,prot_training_steps)

if not os.path.isdir(savedir):
    os.mkdir(savedir)

afile = open(savedir+prefix_AD+'.txt', 'a')
afile.write("%d\t%.8f\t%.8f\n"%(time_steps,avg_entropy_AD,err_entropy_AD))
afile.close()

prefix_th = 'FWD_%ix%ilattice_samples_%i_USINGOPT_theory' %(size,size,samples)

afile = open(savedir+prefix_th+'.txt', 'a')
afile.write("%d\t%.8f\t%.8f\n"%(time_steps,avg_entropy_theory,err_entropy_theory))
afile.close()

print("AD values were: %.8f\t%.8f"%(avg_entropy_AD,err_entropy_AD))
print("Near eqm theory values were: %.8f\t%.8f"%(avg_entropy_theory,err_entropy_theory))

