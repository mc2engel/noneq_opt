
from noneq_opt import barrier_crossing

import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.animation import ArtistAnimation
from matplotlib import rc
from matplotlib import colors
import seaborn as sns
rc('animation', html='jshtml')
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

####load data#####

#parameters to plot:
end_time = float(sys.argv[1]) #128
dt = float(sys.argv[2]) #11 
#degree of Chebyshev polynomials that describe protocols
degree = int(sys.argv[3]) #32
kappa = float(sys.argv[4])
batch_size = int(sys.argv[5]) #256
opt_steps = int(sys.argv[6]) #10 
plot_theory = int(sys.argv[7])

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
kappa_l = kappa/(beta*x_m*2)
kappa_r=kappa_l #pN/nm 
k_trap = 0.4 # pN/nm 
init_position=r0_init*jnp.ones((N,dim))
key = random.PRNGKey(int(time.time()))
key, split1 = random.split(key, 2)

displacement_fn, shift_fn = space.free()
#load files:
loaddir = 'output/'
prefix = 'end_time_%i_dt_%i_chebdeg_%i_kappa_%.3f_batch_%i_trainsteps_%i_' %(end_time, dt,degree,kappa_l,batch_size,opt_steps)

file1 = open(loaddir+prefix+'all_works.pkl', 'rb')
all_works = pickle.load(file1)
file1.close()
file2 = open(loaddir+prefix+'coeffs_.pkl', 'rb')
coeffs_ = pickle.load(file2)
file2.close()

if(plot_theory > 0):
    ###load theoretical protocols###
    theoretical_schedule_2kT = onp.load("../theoretical_protocol_sivak_crooks_2.5kT.npy")
    tvec_2kT=theoretical_schedule_2kT[:,0]
    lambda_t_2kT =theoretical_schedule_2kT[:,1]
    theoretical_schedule_10kT = onp.load("../theoretical_protocol_sivak_crooks_10kT.npy")
    tvec_10kT=theoretical_schedule_10kT[:,0]
    lambda_t_10kT =theoretical_schedule_10kT[:,1]
    
    #make trap fxns:
    theory_trap_fn_2kT = make_theoretical_trap_fxn(jnp.arange(simulation_steps), dt, tvec_2kT, lambda_t_2kT)
    theory_trap_fn_10kT = make_theoretical_trap_fxn(jnp.arange(simulation_steps), dt, tvec_10kT, lambda_t_10kT)
    positions_2kT = theory_trap_fn_2kT(jnp.arange(simulation_steps))
    positions_10kT = theory_trap_fn_10kT(jnp.arange(simulation_steps))


init_coeffs = coeffs_[0]
##### PLOT LOSS AND SCHEDULE EVOLUTION #####
_, (ax0, ax1) = plt.subplots(1, 2, figsize=[24, 12])
barrier_crossing.plot_with_stddev(jnp.array(all_works).T, ax=ax0)
ax0.set_title('Total work')

trap_fn = barrier_crossing.make_trap_fxn(jnp.arange(simulation_steps+1),init_coeffs,r0_init,r0_final)
init_sched = trap_fn(jnp.arange(simulation_steps+1))
ax1.plot(jnp.arange(simulation_steps+1), init_sched, label='initial guess')

for j, coeffs in coeffs_:
  if(j%1 == 0):
    trap_fn = make_trap_fxn(jnp.arange(simulation_steps+1),coeffs,r0_init,r0_final)
    full_sched = trap_fn(jnp.arange(simulation_steps+1))
    ax1.plot(jnp.arange(simulation_steps+1), full_sched, '-', label=f'Step {j}')

#plot final estimate:
trap_fn = make_trap_fxn(jnp.arange(simulation_steps+1),coeffs_[-1][1],r0_init,r0_final)
full_sched = trap_fn(jnp.arange(simulation_steps+1))
ax1.plot(jnp.arange(simulation_steps+1), full_sched, '-', label=f'Final')

if(plot_theory == 1):
    ax1.plot(jnp.arange(simulation_steps),positions_2kT,'--',label='theoretical prediction')
if(plot_theory == 2):
    ax1.plot(jnp.arange(simulation_steps),positions_10kT,'--',label='theoretical prediction')

ax1.legend()#
ax1.set_title('Schedule evolution')
plt.show()
