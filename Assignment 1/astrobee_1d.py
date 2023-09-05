from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
import control


class Astrobee(object):
    def __init__(self,
                 mass=9.6,
                 mass_ac=11.3,
                 h=0.1,
                 **kwargs):
        """
        Astrobee Robot, NMPC tester class.

        :param mass: mass of the Astrobee
        :type mass: float
        :param h: sampling time of the discrete system, defaults to 0.01
        :type h: float, optional
        """

        # Model
        self.n = 2
        self.m = 1
        self.dt = h

        # Model prperties
        self.mass = mass + mass_ac

        # Linearized model for continuous and discrete time
        self.Ac = None
        self.Bc = None
        self.Ad = None
        self.Bd = None

        self.w = 0.0

    def one_axis_ground_dynamics(self):
        """
        Helper function to populate Ac and Bc with continuous-time
        dynamics of the system.
        """

        # Add Ac and Bc matrices
        self.Ac = np.array([[0, 1],[0, 0]])
        self.Bc = np.array([[0, 1/self.mass]]).T

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
        # - the 'ode' contains the equation/function/ode we wish to discretize
        # - the 'p' argument contains the parameters that our function/equation
        #   receives. For now, we will only need the control input u
        ode = {'x': x, 'ode': ca.DM(A) @ x + ca.DM(B) @ u, 'p': ca.vertcat(u)}

        # Here we define the options for our CasADi integrator - it will take care of the
        # numerical integration for us: fear integrals no more!
        options = {"abstol": 1e-5, "reltol": 1e-9, "max_num_steps": 100, "tf": self.dt}

        # Create the integrator
        self.Integrator = ca.integrator('integrator', 'cvodes', ode, options)

        # Now we have an integrator CasADi function. We wish now to take the partial
        # derivaties w.r.t. 'x', and 'u', to obtain Ad and Bd, respectively. That's where
        # we use ca.jacobian passing the integrator we created before - and extracting its
        # value after the integration interval 'xf' (our dt) - and our variable of interest
        Ad = ca.Function('jac_x_Ad', [x, u], [ca.jacobian(
                         self.Integrator(x0=x, p=u)['xf'], x)])
        Bd = ca.Function('jac_u_Bd', [x, u], [ca.jacobian(
                         self.Integrator(x0=x, p=u)['xf'], u)])

        # If you print Ad and Bd, they will be functions that can be evaluated at any point.
        # Now we must extract their value at the linearization point of our chosing!
        x_bar = np.zeros((2, 1))
        u_bar = np.zeros((1, 1))

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

    def set_disturbance(self):
        """
        Activate disturbance acting on the system
        """
        self.w = -0.002

    def disable_disturbance(self):
        """
        Disable the disturbance effect.
        """
        self.w = 0.0

    def get_disturbance(self):
        """
        Return the disturbance value

        :return: disturbance value
        :rtype: float
        """
        return self.w

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

        if self.w != 0.0:
            Bw = ca.DM.zeros(2, 1)
            Bw[1, 0] = 1
            x_next = x_next - Bw * self.w

        return x_next

    def poles_zeros(self, Ad, Bd, Cd, Dd):
        """
        Plots the system poles and zeros.

        :param Ad: state transition matrix
        :type Ad: np.ndarray
        :param Bd: control matrix
        :type Bd: np.ndarray
        :param Cd: state-observation matrix
        :type Cd: np.ndarray
        :param Dd: control-observation matrix
        :type Dd: np.ndarray
        """
        # dt == 0 -> Continuous time system
        # dt != 0 -> Discrete time system
        sys = control.StateSpace(Ad, Bd, Cd, Dd, dt=self.dt)
        control.pzmap(sys)
        plt.show()
        return
