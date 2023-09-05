import numpy as np
import control
import astrobee_1d
import controller

# Create an Astrobee instance
Astrobee = astrobee_1d.Astrobee() 

# Create the A, B, C and D matrices
A, B = Astrobee.one_axis_ground_dynamics()
C = np.array([[1, 0]])
D = np.array([[0]])

# Discretize by using casadi_c2d (the differences are marginal)
Ad, Bd, Cd, Dd = Astrobee.casadi_c2d(A, B, C, D)

# Create two StateSpace systems: one continuous, one discrete
ContSys = control.StateSpace(A, B, C, D)
ContPoles = control.poles(ContSys)
ContZeros = control.zeros(ContSys)
print(f"Continuous system has poles: {ContPoles} and zeros: {ContZeros}")

DiscSys = control.StateSpace(Ad, Bd, Cd, Dd, Astrobee.dt)
DiscPoles = control.poles(DiscSys)
DiscZeros = control.zeros(DiscSys)
print(f"Discrete system has poles: {DiscPoles} and zeros: {DiscZeros}")

# Q4: Calculate the controller gains L for different poles
Ctrl = controller.Controller()
Ctrl.set_system(Ad, Bd, Cd, Dd)
L = Ctrl.get_closed_loop_gain(np.exp(np.array([-0.02-0.005j, -0.02+0.005j])).tolist())
print(L)
