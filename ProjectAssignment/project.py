import numpy as np
import yaml

from astrobee import Astrobee
from mpc import MPC
from simulation import EmbeddedSimEnvironment
import user_settings

PARAMETER_FILE = 'P2'

# TODO: Set the path to the trajectory file:
#       eg.: trajectory_quat = '/home/roque/Project Assignment/Dataset/trajectory_quat.txt'
trajectory_quat = user_settings.trajectory_quat

# TODO: complete the 'tuning_file_path' variable to the path of your tuning.yaml
#       eg.: tuning_file_path = '/home/roque/Project Assignment/tuning.yaml'
tuning_file_path = user_settings.tuning_file_path

# Q1
# TODO: Set the Astrobee dynamics in Astrobee->astrobee_dynamics_quat
abee = Astrobee(trajectory_file=trajectory_quat)

# If successful, test-dynamics should not complain
# abee.test_dynamics()

# Instantiate controller
u_lim, x_lim = abee.get_limits()

# Create MPC Solver
# TODO: Select the parameter type with the argument param='P1'  - or 'P2', 'P3'
MPC_HORIZON = 8
with open(tuning_file_path, 'r') as stream:
    parameters = yaml.safe_load(stream)
    solver_opts = {
        'ipopt.print_level': 0,
        'ipopt.max_iter': parameters[PARAMETER_FILE]['max_iter'],
        'ipopt.tol': parameters[PARAMETER_FILE]['tol'],
    }
# ctl = MPC(model=abee,
#           dynamics=abee.model,
#           param='P2',
#           N=MPC_HORIZON,
#           ulb=-u_lim, uub=u_lim,
#           xlb=-x_lim, xub=x_lim,
#           tuning_file=tuning_file_path)

# Q2: Reference tracking
# TODO: adjust the tuning.yaml parameters for better performance
# x_d = abee.get_static_setpoint()
# ctl.set_reference(x_d)
# Set initial state
x0 = abee.get_initial_pose()
# sim_env = EmbeddedSimEnvironment(model=abee,
#                                  dynamics=abee.model,
#                                  controller=ctl.mpc_controller,
#                                  time=80)
# t, y, u = sim_env.run(x0)
# sim_env.visualize()  # Visualize state propagation

# Q3: Activate Tracking
# TODO: complete the MPC class for reference tracking
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
# t, y, u = sim_env_tracking.run(x0)A
# sim_env_tracking.visualize()  # Visualize state propagation
# sim_env_tracking.visualize_error()


# Test 3: Activate forward propagation
# TODO: complete the MPC Astrobee class to be ready for forward propagation
# abee.test_forward_propagation()
tracking_ctl.set_forward_propagation()
t, st, y, u, e = sim_env_tracking.run(x0)
sim_env_tracking.calcScore()
sim_env_tracking.visualize()  # Visualize state propagation
sim_env_tracking.visualize_error()
sim_env_tracking.plot3DTrajectory()
sim_env_tracking.plot3DTrajectoryWithBoxes()
