import numpy as np

from astrobee import Astrobee
from dlqr import DLQR
from simulation import EmbeddedSimEnvironment

import time

def runSim(Q_c, R_c, ctl, abee, x0):
    Q = np.diag(Q_c)
    R = np.diag(R_c)
    ctl.get_lqr_gain(Q, R)
    sim_env = EmbeddedSimEnvironment(model=abee,
                                     dynamics=abee.linearized_discrete_dynamics,
                                     controller=ctl.feedback,
                                     time=20)
    t, y, u = sim_env.run(x0, plot=False)
    #sim_env.evaluate_performance(t, y, u)
    return sim_env.eval_perf(t, y, u)



# Init system
abee = Astrobee(h=0.1)

# Linearization around reference point
x_star = np.zeros((12, 1))
x_star[0] = 1
x_star[1] = 0.5
x_star[2] = 0.1
x_star[6] = 0.087
x_star[7] = 0.077
x_star[8] = 0.067
x0 = np.zeros((12, 1))

# Linearize
A, B = abee.create_linearized_dynamics(x_bar=x_star)
C = np.diag(np.ones(12))
D = np.zeros((12, 6))
Ad, Bd, Cd, Dd = abee.casadi_c2d(A, B, C, D)
ctl = DLQR(Ad, Bd, C)
abee.set_discrete_dynamics(Ad, Bd)

# Define coefficients
R_c = np.ones(6)
Q_c = np.ones(12)

coeff = [i*i for i in np.linspace(1,9,8)]
ctl.set_reference(x_star)

# Grid Search
# for k01 in range(len(coeff)):
#     Q_c[0:3] = coeff[k01]
#     for k02 in range(len(coeff)):
#         Q_c[3:6] = coeff[k02]
#         for k03 in range(len(coeff)):
#             print(f'Loops at {k01}, {k02}, {k03}')
#             start = time.time()
#             Q_c[6:9] = coeff[k03]
#             for k04 in range(len(coeff)):
#                 Q_c[9:12] = coeff[k04]
#                 for k05 in range(len(coeff)):
#                     R_c[0:3] = coeff[k05]
#                     for k06 in range(len(coeff)):
#                         R_c[3:6] = coeff[k06]
#                         if runSim(Q_c, R_c, ctl, sim_env, x0): 
#                             print(f'DONE with: Q_c {Q_c} R_c {R_c}')
#             stop = time.time()
#             elapsed = stop - start
#             full = elapsed * len(coeff) * len(coeff) * len(coeff)
#             print(f"After {elapsed}s, estimate total is {full/60/60}h")

# Random Search
for i in range(10000000):
    Q_c[0:3] = np.random.randint(1, 200, 3)
    Q_c[3:6] = np.random.randint(1, 100)
    Q_c[6:9] = np.random.randint(1, 20)
    Q_c[9:12] = np.random.randint(1, 10)
    R_c[0:3] = np.random.randint(1, 200, 3)
    R_c[3:6] = np.random.randint(1, 50)

    if runSim(Q_c, R_c, ctl, abee, x0):
        print(f'DONE! Found valid configuration: Q_c {Q_c} R_c {R_c}')
        break
    if np.mod(i, 10000) == 0:
        print(f'Running config {i/1000}k')