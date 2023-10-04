import numpy as np
import yaml

from astrobee import Astrobee
from mpc import MPC
from simulation import EmbeddedSimEnvironment
import user_settings

PARAMETER_FILE = 'P1'
trajectory_quat = user_settings.trajectory_quat
tuning_file_path = user_settings.tuning_file_path

# Define Astrobee
abee = Astrobee(trajectory_file=trajectory_quat)
u_lim, x_lim = abee.get_limits()

# Define Parameters
MPC_HORIZON = 8
with open(tuning_file_path, 'r') as stream:
    parameters = yaml.safe_load(stream)
    solver_opts = {
        'ipopt.print_level': 0,
        'ipopt.max_iter': parameters[PARAMETER_FILE]['max_iter'],
        'ipopt.tol': parameters[PARAMETER_FILE]['tol'],
    }

x0 = abee.get_initial_pose()

# Create Controller and Sim
tracking_ctl = MPC(model=abee,
                   dynamics=abee.model,
                   param=PARAMETER_FILE,
                   N=MPC_HORIZON,
                   trajectory_tracking=True,
                   ulb=-u_lim, uub=u_lim,
                   xlb=-x_lim, xub=x_lim,
                   tuning_file=tuning_file_path,
                   solver_opts=solver_opts)
sim_env_tracking = EmbeddedSimEnvironment(model=abee,
                                          dynamics=abee.model,
                                          controller=tracking_ctl.mpc_controller,
                                          time=80)

# Run
tracking_ctl.set_forward_propagation()
t, st, y, u, e = sim_env_tracking.run(x0)
sim_env_tracking.calcScore()
sim_env_tracking.visualize()  # Visualize state propagation
# sim_env_tracking.visualize_error()
