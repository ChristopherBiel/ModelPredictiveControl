import numpy as np

from astrobee import Astrobee
from dlqr import DLQR
from simulation import EmbeddedSimEnvironment

import time

# This script performs hyperparameter optimization
# by using random search for specific values of Q_c and R_c
# Automatic evaluation then allows for selection of
# the best performing configurations

def runSim(Q_c, R_c, ctl, abee, x0):
    '''
    Helper function to run and evaluate a single combination of
    R_c and Q_c. We have to createt a new SimEnvironment, since
    the parameters of the LQR are not adapted when re-executing
    get_lqr_gain for an existing SimEnv
    '''

    Q = np.diag(Q_c)
    R = np.diag(R_c)
    ctl.get_lqr_gain(Q, R)
    sim_env = EmbeddedSimEnvironment(model=abee,
                                     dynamics=abee.linearized_discrete_dynamics,
                                     controller=ctl.feedback,
                                     time=20)
    t, y, u = sim_env.run(x0, plot=False)

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

# Grid search is a possibility, but usually has inferior performance
# when compared to Random Search
lowest_cost = 100
config = [Q_c, R_c]
# The script is supposed to run 'forever' (approx. 12h on my hardware)
# but can be stopped at any time
# The user can then use the latest result as read from the terminal
for i in range(10000000):
    # The min/max values for the random distribution were chosen based
    # on manual testing and the first test results of the random search
    Q_c[0:3] = np.random.randint(1, 300, 3)
    Q_c[3:6] = np.random.randint(1, 200, 3)
    Q_c[6:9] = np.random.randint(1, 50)
    Q_c[9:12] = np.random.randint(1, 30)
    R_c[0:3] = np.random.randint(1, 300, 3)
    R_c[3:6] = np.random.randint(1, 200)
    curr_cost = runSim(Q_c, R_c, ctl, abee, x0)
    if curr_cost < lowest_cost:
        config = [Q_c, R_c]
        lowest_cost = curr_cost
        print(f'At {i}: New lowest cost {lowest_cost} with Q_c: {Q_c}, R_c: {R_c}')