import jax
import jax.numpy as jnp
import jax.example_libraries.optimizers as jopt
import numpy as np

import os
import functools
import pickle

import tqdm
import time

from noneq_opt import ising
from noneq_opt import parameterization as p10n

import sys

##simulation and training parameters
size = int(sys.argv[1]) #32
seed = int(sys.argv[2]) #0 

time_steps = int(sys.argv[3]) #51 

#degree of Chebyshev polynomials that describe protocols
field_degree = log_temp_degree= int(sys.argv[4]) #32
batch_size = int(sys.argv[5]) #256
training_steps = int(sys.argv[6]) #5
save_every = int(sys.argv[7]) #1
init=0.1
lr = jopt.exponential_decay(init, training_steps, .01)
optimizer = jopt.adam(lr)
max_grad_ = 1
print("Running Ising optimization for ",size,"x",size," lattice for ",time_steps," time steps.")

#Define initial guess for the optimal protocol
#We do this by defining "baseline" functions and learning a "diff" from these baselines.

Tbaseline= ising.log_temp_baseline(min_temp=0.65)
Fbaseline = ising.field_baseline()

init_guess_log_temp =  jnp.zeros(log_temp_degree)
init_guess_field =  jnp.zeros(field_degree)


initial_log_temp_schedule = p10n.AddBaseline(
    p10n.ConstrainEndpoints(
        p10n.Chebyshev(
            init_guess_log_temp
        ),
        y0=0.,
        y1=0.,
    ),
    baseline=Tbaseline
)


initial_field_schedule = p10n.AddBaseline(
    p10n.ConstrainEndpoints(
        p10n.Chebyshev(
            init_guess_field
        ),
        y0=0.,
        y1=0.,
    ),
    baseline=Fbaseline
)

assert initial_field_schedule.domain == (0., 1.)
assert initial_log_temp_schedule.domain == (0., 1.)

initial_schedule = schedule = ising.IsingSchedule(
    initial_log_temp_schedule, initial_field_schedule)

stream = ising.seed_stream(0)
state = optimizer.init_fn(schedule)
initial_spins = -jnp.ones([size, size]) #start with all spins 'up'

train_step = ising.get_train_step(optimizer,
                                  initial_spins,
                                  batch_size,
                                  time_steps,
                                  ising.total_entropy_production,
                                  mode = 'fwd',
                                  max_grad = max_grad_)
from jax.lib import xla_bridge
print("using backend: ",xla_bridge.get_backend().platform)
summaries = []
entropies = []
log_temp = []
fields = []
log_temp_weights = []
fields_weights = []
gradients = []
grad_REINFORCE_term = []
grad_traj_term = []
times = jnp.linspace(0, 1, time_steps)
initial_spin_state = ising.IsingState(initial_spins, ising.map_slice(initial_schedule(times), 0))
log_p_init = - ising.energy(initial_spin_state) / jnp.exp(initial_schedule(times).log_temp[0])
final_kT = jnp.exp(initial_schedule(times).log_temp[-1])

for j in tqdm.trange(training_steps, position=0):
  state, summary = train_step(state, j, next(stream))
  #add correction term:
  log_p_final = - summary.energy[:,-1] / final_kT
  correction = log_p_init - log_p_final
  final_entropies = np.array(jax.device_get(summary.entropy_production))
  final_entropies[:,-1] = final_entropies[:,-1] + correction
  entropies.append(final_entropies) 
  if((j%save_every == 0) or (j==(training_steps-1))):
    print("saving iteration ",j)
    summaries.append(jax.device_get(summary))
    log_temp.append(optimizer.params_fn(state).log_temp)
    log_temp_weights.append(optimizer.params_fn(state).log_temp.variables['wrapped'].variables['wrapped'].weights)
    fields.append(optimizer.params_fn(state).field)
    fields_weights.append(optimizer.params_fn(state).field.variables['wrapped'].variables['wrapped'].weights)

#save output:
savedir = 'output/'
prefix = '%ix%ilattice_timesteps_%i_chebdeg_%i_batch_%i_trainsteps_%i_' %(size,size,time_steps,field_degree,batch_size,training_steps)

if not os.path.isdir(savedir):
    os.mkdir(savedir)

#note that you cannot pickle locally-defined functions (like the log_temp_baseline() thing), so we save the Chebyshev weights and then later recreate the schedules
afile = open(savedir+prefix+'summaries.pkl', 'wb')
pickle.dump(summaries, afile)
afile.close()

afile = open(savedir+prefix+'entropies.pkl', 'wb')
pickle.dump(entropies, afile)
afile.close()

afile = open(savedir+prefix+'log_temp_weights.pkl', 'wb')
pickle.dump(log_temp_weights, afile)
afile.close()

afile = open(savedir+prefix+'field_weights.pkl', 'wb')
pickle.dump(fields_weights, afile)
afile.close()
