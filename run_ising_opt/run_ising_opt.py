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

##simulation and training parameters
size = 128
seed = 0 

time_steps = 11 

#degree of Chebyshev polynomials that describe protocols
field_degree = log_temp_degree= 32
batch_size = 256
training_steps = 10
save_every = 1 
lr = jopt.exponential_decay(.1, .9 * training_steps, 0.1)
optimizer = jopt.adam(lr)

#Define initial guess for the optimal protocol
#We do this by defining "baseline" functions and learning a "diff" from these baselines.
def log_temp_baseline(min_temp=.69, max_temp=10., degree=2):
  def _log_temp_baseline(t):
    scale = (max_temp - min_temp)
    shape = (1 - t)**degree * t**degree * 4 ** degree
    return jnp.log(shape * scale + min_temp)
  return _log_temp_baseline

def field_baseline(start_field=1., end_field=-1.):
  def _field_baseline(t):
    return (1 - t) * start_field + t * end_field
  return _field_baseline

Tbaseline=log_temp_baseline()
Fbaseline = field_baseline()

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
initial_spins = jnp.ones([size, size]) #start with all spins 'up'

train_step = ising.get_train_step(optimizer,
                                  initial_spins,
                                  batch_size,
                                  time_steps,
                                  ising.total_entropy_production,
                                  mode = 'fwd')

summaries = []
entropies = []
log_temp = []
fields = []
log_temp_weights = []
fields_weights = []
gradients = []
grad_REINFORCE_term = []
grad_traj_term = []
for j in tqdm.trange(training_steps, position=0):
  state, summary = train_step(state, j, next(stream))
  entropies.append(jax.device_get(summary.entropy_production))
  if(j%save_every == 0):
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
