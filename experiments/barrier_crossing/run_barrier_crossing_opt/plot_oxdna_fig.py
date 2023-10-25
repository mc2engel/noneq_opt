import matplotlib.pyplot as plt
from noneq_opt import barrier_crossing
import numpy as np
import pickle
import pdb 
import seaborn as sns
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.ticker as ticker
import sys 

def plot_with_stddev(x, label=None, n=1, axis=0, ax=plt):
  stddev = np.std(x, axis)
  mn = np.mean(x, axis)

  ax.fill_between(np.arange(mn.shape[0]),
                  mn + n * stddev, mn - n * stddev, alpha=.3)
  ax.plot(mn, '-o', label=label)


#plot params
fontsize=26
plt.rcParams['axes.linewidth'] = 1 
insetlabels=16
lincolor = 'tab:grey'
ADcolor= 'tab:orange'
thcolor= 'tab:cyan'


#save_filepath = '/content/drive/MyDrive/Colab Notebooks/Barrier_crossing/Data_for_paper/2.5kT_barrier/'

#num_curves = 6
#baseline_palette = sns.color_palette('YlGnBu', n_colors=num_curves+1)[1:]
#theory_colour = sns.xkcd_palette(['burnt orange'] * 2)
#sns.palplot(baseline_palette + theory_colour[:1])

#palette_ = baseline_palette + theory_colour[:1]

#parameters to plot:
end_time = float(sys.argv[1]) #128
dt = float(sys.argv[2]) #11 
#degree of Chebyshev polynomials that describe protocols
degree = int(sys.argv[3]) #32
kappa = float(sys.argv[4])
batch_size = int(sys.argv[5]) #256
opt_steps = int(sys.argv[6]) #10 

simulation_steps = int((end_time)/dt)
r0_init = 0.
r0_final = 40.
init_coeffs = np.array([2.00000000e+01,  1.99599600e+01, -2.63245988e-15,  8.41484236e-16,
  0.00000000e+00, -7.47209997e-15, -1.69086360e-14,  1.27892999e-14,
  3.02606960e-15, -6.05805446e-15, -6.36600761e-16, -5.53701086e-15,
 -1.14938537e-14])#end time 1e-6
#init_coeffs = np.array([2.00000000e+01,  1.95959596e+01,  2.08860886e-15,  3.10696421e-14,
# -1.67759313e-14, -1.52606588e-14, -9.98246452e-15, -5.88161754e-15,
#  1.67185032e-14,  1.21238201e-14,  5.74563866e-15,  6.62046515e-15,
# -6.89056561e-15])#end tme 1e-7 
loaddir = 'output/end_time_%.10f_dt_%.10f_chebdeg_%i_kappa_%.4f_batch_%i_trainsteps_%i_' %(end_time, dt,degree,kappa,batch_size,opt_steps)

file2 = open(loaddir+'coeffs.pkl', 'rb')
coeffs_load = pickle.load(file2)
file2.close()

file2 = open(loaddir+'all_works.pkl', 'rb')
works_load = pickle.load(file2)
file2.close()

##### PLOT LOSS  #####
fig, (ax0, ax1) = plt.subplots(1, 2, figsize=[24, 12])
plot_with_stddev(np.array(works_load).T, ax=ax0)
#ax0.set_title('Total work')
ax0.set_xlabel('Optimization iteration',fontsize=fontsize)
ax0.set_ylabel('Dissipated Work (kT)',fontsize=fontsize)
ax0.tick_params(axis='x', labelsize=fontsize)
ax0.tick_params(axis='y', labelsize=fontsize)


##### PLOT SCHEDULE EVOLUTION  #####
trap_fn = barrier_crossing.make_trap_fxn(np.arange(simulation_steps+1),init_coeffs,r0_init,r0_final)
init_sched = trap_fn(np.arange(simulation_steps+1))
ax1.plot(0.001*np.arange(simulation_steps+1), init_sched, label='initial guess')
for j, coeffs in coeffs_load:
   if((j-1)%100 == 0):
     trap_fn = barrier_crossing.make_trap_fxn(np.arange(simulation_steps+1),coeffs,r0_init,r0_final)
     full_sched = trap_fn(np.arange(simulation_steps+1))
     if(j==0):
       ax1.plot(0.001*np.arange(simulation_steps+1), full_sched, '-', label=f'Step {j}') 
     else:
       ax1.plot(0.001*np.arange(simulation_steps+1), full_sched, '-', label=f'Step {j-1}')
trap_fn = barrier_crossing.make_trap_fxn(np.arange(simulation_steps+1),coeffs_load[-1][1],r0_init,r0_final)
full_sched = trap_fn(np.arange(simulation_steps+1))
ax1.plot(0.001*np.arange(simulation_steps+1), full_sched, '-', label='Final')
ax1.set_xlabel("Simulation time (microseconds)",fontsize=fontsize)
ax1.set_ylabel("Trap position (nm)",fontsize=fontsize)
ax1.tick_params(axis='x', labelsize=fontsize)
ax1.tick_params(axis='y', labelsize=fontsize)
fig.text(0.1, 0.895, '(a)', va='top', ha='left', fontsize=fontsize)
fig.text(0.5, 0.895, '(b)', va='top', ha='left', fontsize=fontsize)
ax1.legend(fontsize=16)
fig.savefig("1us_oxdna_parallel_k1.5.pdf",format='pdf')
fig.show()
