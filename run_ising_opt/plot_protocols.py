import numpy as np
import matplotlib.pyplot as plt
from noneq_opt import ising
from noneq_opt import parameterization as p10n
import pickle
import sys

def plot_schedules(schedules, time_steps):
  times = np.linspace(0, 1, 100)
  fig, ax = plt.subplots(1, 3, figsize=[21, 6])
  for name, sched in schedules.items():
    temp = np.exp(sched.log_temp(times))
    field = sched.field(times)
    ax[0].plot(time_steps*times, temp, label=name)
    ax[1].plot(time_steps*times, field, label=name)
    ax[2].plot(temp, field, label=name)

  ax[0].set_title('Time vs. Temperature')
  ax[0].set_xlabel('Time')
  ax[0].set_ylabel('Temperature')
  ax[0].legend()

  ax[1].set_title('Time vs. Field')
  ax[1].set_xlabel('Time')
  ax[1].set_ylabel('Field')
  ax[1].legend()

  ax[2].set_title('Temperature vs. Field')
  ax[2].set_xlabel('Temperature')
  ax[2].set_ylabel('Field')
  ax[2].legend()


#parameters to plot:
size = int(sys.argv[1]) #128
time_steps = int(sys.argv[2]) #11 
#degree of Chebyshev polynomials that describe protocols
field_degree = log_temp_degree= int(sys.argv[3]) #32
batch_size = int(sys.argv[4]) #256
training_steps = int(sys.argv[5]) #10 
skip_every = int(sys.argv[6])

#load files:
loaddir = '../launch_scripts/output/'
prefix = '%ix%ilattice_timesteps_%i_chebdeg_%i_batch_%i_trainsteps_%i_' %(size,size,time_steps,field_degree,batch_size,training_steps)

file1 = open(loaddir+prefix+'log_temp_weights.pkl', 'rb')
log_temp_weights_load = pickle.load(file1)
file1.close()

file2 = open(loaddir+prefix+'field_weights.pkl', 'rb')
field_weights_load = pickle.load(file2)
file2.close()


#redefine schedules from loaded data:
schedules = []
Tbaseline = ising.log_temp_baseline() 
Fbaseline = ising.field_baseline()

for i in range(len(log_temp_weights_load)):
  log_temp_schedule = p10n.AddBaseline(
      p10n.ConstrainEndpoints(
          p10n.Chebyshev(
              log_temp_weights_load[i]
          ),
          y0=0.,
          y1=0.,
      ),
      baseline=Tbaseline
  )

  field_schedule = p10n.AddBaseline(
      p10n.ConstrainEndpoints(
          p10n.Chebyshev(
              field_weights_load[i]
          ),
          y0=0.,
          y1=0.,
      ),
      baseline=Fbaseline
  )
  assert field_schedule.domain == (0., 1.)
  assert log_temp_schedule.domain == (0., 1.)
  schedules.append(ising.IsingSchedule(
      log_temp_schedule, field_schedule))

my_dict={}
print("num scheds saved: ",len(schedules))
for j in range(len(schedules)):
    #print(j)
    if(j%skip_every==0):
        my_dict[f'Step {j}']=schedules[j]

plot_schedules(my_dict,time_steps)
plt.show()
