from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import casadi as ca
import numpy as np


class Astrobee(object):
    def __init__(self,
                 mass=9.6,
                 mass_ac=11.3,
                 inertia=0.25,
                 h=0.1,
                 **kwargs):
        """
        Astrobee Robot, NMPC tester class.

        :param mass: mass of the Astrobee
        :type mass: float
        :param inertia: inertia tensor of the Astrobee
        :type inertia: np.diag
        :param h: sampling time of the discrete system, defaults to 0.01
        :type h: float, optional
        """

        # Model
        self.n = None
        self.m = None
        self.dt = h

        # Model prperties
        self.mass = mass + mass_ac
        self.inertia = inertia

        # Linearized model for continuous and discrete time
        self.Ac = None
        self.Bc = None
        self.Ad = None
        self.Bd = None

        # Set single agent properties
        self.set_casadi_options()

    def set_casadi_options(self):
        """
        Helper function to set casadi options.
        """
        self.fun_options = {
            "jit": False,
            "jit_options": {"flags": ["-O2"]}
        }

    def cartesian_ground_dynamics(self):
        """
        Helper function to populate Ac and Bc with continuous-time
        dynamics of the system.
        """

        # Jacobian of exact discretization
        self.n = 6
        self.m = 3

        Ac = ca.DM.zeros(self.n, self.n)
        Bc = ca.DM.zeros(self.n, self.m)

        # TODO: Fill the matrices Ac and Bc according to the model in (1)

        self.Ac = np.asarray(Ac)
        self.Bc = np.asarray(Bc)

        return self.Ac, self.Bc

    def cartesian_3d_dynamics(self):
        """
        Helper function to populate Ac and Bc with continuous-time
        dynamics of the system.
        """

        # Jacobian of exact discretization
        self.n = 8
        self.m = 4

        Ac = ca.DM.zeros(self.n, self.n)
        Bc = ca.DM.zeros(self.n, self.m)

        # TODO: Fill the matrices Ac and Bc according to the model in (1), adding
        #       the proper component for translation on Z

        self.Ac = np.asarray(Ac)
        self.Bc = np.asarray(Bc)

        return self.Ac, self.Bc

    def linearized_dynamics(self, x, u):
        """
        Linear dynamics for the Astrobee, continuous time.

        :param x: state
        :type x: np.ndarray, ca.DM, ca.MX
        :param u: control input
        :type u: np.ndarray, ca.DM, ca.MX
        :return: state derivative
        :rtype: np.ndarray, ca.DM, ca.MX
        """

        xdot = self.Ac @ x + self.Bc @ u

        return xdot

    def casadi_c2d(self, A, B, C, D):
        """
        Continuous to Discrete-time dynamics
        """
        # Set CasADi variables
        x = ca.MX.sym('x', A.shape[1])
        u = ca.MX.sym('u', B.shape[1])

        # Create an ordinary differential equation dictionary. Notice that:
        # - the 'x' argument is the state
        # - the 'ode' contains the equation/function we wish to discretize
        # - the 'p' argument contains the parameters that our function/equation
        #   receives. For now, we will only need the control input u
        ode = {'x': x, 'ode': ca.DM(A) @ x + ca.DM(B) @ u, 'p': ca.vertcat(u)}

        # Here we define the options for our CasADi integrator - it will take care of the
        # numerical integration for us: fear integrals no more!
        options = {"abstol": 1e-5, "reltol": 1e-9, "max_num_steps": 100, "tf": self.dt}

        # Create the integrator
        self.Integrator = ca.integrator('integrator', 'cvodes', ode, options)

        # Now we have an integrator CasADi function. We wish now to take the partial
        # derivaties w.r.t. 'x', and 'u', to obtain Ad and Bd, respectively. That's wher
        # we use ca.jacobian passing the integrator we created before - and extracting its
        # value after the integration interval 'xf' (our dt) - and our variable of interest
        Ad = ca.Function('jac_x_Ad', [x, u], [ca.jacobian(
                         self.Integrator(x0=x, p=u)['xf'], x)])
        Bd = ca.Function('jac_u_Bd', [x, u], [ca.jacobian(
                         self.Integrator(x0=x, p=u)['xf'], u)])

        # If you print Ad and Bd, they will be functions that can be evaluated at any point.
        # Now we must extract their value at the linearization point of our chosing!
        x_bar = np.zeros((self.n, 1))
        u_bar = np.zeros((self.m, 1))

        return np.asarray(Ad(x_bar, u_bar)), np.asarray(Bd(x_bar, u_bar)), C, D

    def set_discrete_dynamics(self, Ad, Bd):
        """
        Helper function to populate discrete-time dynamics

        :param Ad: discrete-time transition matrix
        :type Ad: np.ndarray, ca.DM
        :param Bd: discrete-time control input matrix
        :type Bd: np.ndarray, ca.DM
        """

        self.Ad = Ad
        self.Bd = Bd

    def linearized_discrete_dynamics(self, x, u):
        """
        Method to propagate discrete-time dynamics for Astrobee

        :param x: state
        :type x: np.ndarray, ca.DM
        :param u: control input
        :type u: np.ndarray, ca.DM
        :return: state after dt seconds
        :rtype: np.ndarray, ca.DM
        """

        if self.Ad is None or self.Bd is None:
            print("Set discrete-time dynamics with set_discrete_dynamcs(Ad, Bd) method.")
            return np.zeros(x.shape[0])

        x_next = self.Ad @ x + self.Bd @ u

        return x_next

    def set_trajectory(self, time, type="2d", x_off=np.zeros((4, 1)), fp=0.1, ft=0.01):
        """
        Helper methjod to create a trajectory for Astrobee to perform in open-loop

        :param time: total length of the trajectory
        :type time: float
        """
        t = np.linspace(0, time, int(time / self.dt))
        px = 0.025 * np.cos(2 * np.pi * fp * t) + x_off[0]  # 0.05 * np.ones(t.shape)
        py = 0.025 * np.sin(2 * np.pi * fp * t) + x_off[1]  # np.zeros(t.shape)
        theta = 0.05 * np.cos(2 * np.pi * ft * t + x_off[3])  # np.zeros(t.shape)
        if type != "2d":
            pz = 0.025 * np.cos(2 * np.pi * fp * t) + x_off[2]
            self.trajectory = np.vstack((px, py, pz, theta))
        else:
            self.trajectory = np.vstack((px, py, theta))

    def get_trajectory(self, t_start, t_end=None):
        """
        Get part of the trajectory created previously.

        :param t_start: starting time
        :type t_start: float
        :param t_end: ending time, defaults to None
        :type t_end: float, optional
        """

        start_idx = int(t_start / self.dt)

        if t_end is None:
            piece = self.trajectory[:, start_idx:]
        else:
            end_idx = int(t_end / self.dt)
            piece = self.trajectory[:, start_idx:end_idx]

        return piece
