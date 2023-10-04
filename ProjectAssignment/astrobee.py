from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import casadi as ca
import numpy as np
import numpy.matlib as nmp
from util import *


class Astrobee(object):
    def __init__(self,
                 trajectory_file,
                 mass=9.6,
                 inertia=np.diag([0.1534, 0.1427, 0.1623]),
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
        :param model: select between 'euler' or 'quat'
        :type model: str
        """

        # Model
        self.nonlinear_model = self.astrobee_dynamics_quat
        self.n = 13
        self.m = 6
        self.dt = h

        # Model prperties
        self.mass = mass
        self.inertia = inertia

        # Set CasADi functions
        self.set_casadi_options()

        # Set nonlinear model with a RK4 integrator
        self.model = self.rk4_integrator(self.nonlinear_model)

        # Set path for trajectory file
        self.trajectory_file = trajectory_file

    def set_casadi_options(self):
        """
        Helper function to set casadi options.
        """
        self.fun_options = {
            "jit": False,
            "jit_options": {"flags": ["-O2"]}
        }

    def astrobee_dynamics_quat(self, x, u):
        """
        Astrobee nonlinear dynamics with Quaternions.

        :param x: state
        :type x: ca.MX
        :param u: control input
        :type u: ca.MX
        :return: state time derivative
        :rtype: ca.MX
        """

        # State extraction
        p = x[0:3]
        v = x[3:6]
        q = x[6:10]
        w = x[10:]

        # 3D Force
        f = u[0:3]

        # 3D Torque
        tau = u[3:]

        # Model
        pdot = v
        vdot = ca.mtimes(r_mat_q(q), f) / self.mass
        qdot = ca.mtimes(xi_mat(q), w) / 2
        wdot = ca.mtimes(ca.inv(self.inertia), tau - ca.mtimes(skew(w),
                         ca.mtimes(self.inertia, w)))

        dxdt = [pdot, vdot, qdot, wdot]

        return ca.vertcat(*dxdt)

    def rk4_integrator(self, dynamics):
        """
        Runge-Kutta 4th Order discretization.
        :param x: state
        :type x: ca.MX
        :param u: control input
        :type u: ca.MX
        :return: state at next step
        :rtype: ca.MX
        """
        x0 = ca.MX.sym('x0', self.n, 1)
        u = ca.MX.sym('u', self.m, 1)

        x = x0

        k1 = dynamics(x, u)
        k2 = dynamics(x + self.dt / 2 * k1, u)
        k3 = dynamics(x + self.dt / 2 * k2, u)
        k4 = dynamics(x + self.dt * k3, u)
        xdot = x0 + self.dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

        # Normalize quaternion: TODO(Pedro-Roque): check best way to propagate
        rk4 = ca.Function('RK4', [x0, u], [xdot], self.fun_options)

        return rk4

    def forward_propagate(self, x_s, npoints, radius=0.5):
        """
        Forward propagate the observed state given a constant velocity.

        The output should be a self.n x npoints matrix, with the
        desired offset track.

        :param x_s: starting state
        :type x_s: np.ndarray
        :param npoints: number of points to propagate
        :type npoints: int
        :return: forward propagated trajectory
        :rtype: np.ndarray
        """

        x_r = np.zeros((self.n, npoints))

        # TODO: do the forward propagation of the measured state x_s for npoints.
        # Ensure the input has the right dimensions!
        # The implementation in get_trajectory provides this function
        # With x_s in shape (13,) while the test provides it as (12,1)
        if x_s.ndim == 2 and x_s.shape[1] == 1:
            x_s = x_s.ravel()
            print(f"Adjusted shape to {x_s.shape}")
        
        # State extraction
        p = x_s[0:3]
        v = x_s[3:6]
        q = x_s[6:10]
        w = x_s[10:]
        # Calculate the reference position for the t=0 entry
        p_r = p + r_mat_q_np(q)[:,0]*radius
        x_r[:,0] = np.concatenate((p_r, v, q, w), axis=0)
    
        for t in range(npoints-1):
            # Forward propagate honeys movement
            p += v * self.dt
            q += np.dot(xi_mat_np(q),w) * self.dt / 2
            q = q / np.linalg.norm(q)
            # Calculate the reference position
            p_r = p + r_mat_q_np(q)[:,0]*radius
            # Fill x_r
            x_r[:,t+1] = np.concatenate((p_r, v, q, w), axis=0)
        return x_r

    def get_trajectory(self, t, npoints, forward_propagation=False):
        """
        Provide trajectory to be followed.
        :param t0: starting time
        :type t0: float
        :param npoints: number of trajectory points
        :type npoints: int
        :return: trajectory with shape (Nx, npoints)
        :rtype: np.array
        """

        if t == 0.0:
            # Load the trajectory from the txt file
            tmp = np.loadtxt(self.trajectory_file, ndmin=2)
            self.trajectory = tmp.reshape((self.n, int(tmp.shape[0] / self.n)), order="F")
            # Trajectory is of shape 13x14715 with the max. values of
            # [ 11 -7  4.8  0.00068 0.00054  0.0012  0.718  0.784 
            # 0.65  1  0.0548  0.0365 0.0067]

        if forward_propagation is False:
            id_s = int(round(t / self.dt))
            id_e = int(round(t / self.dt)) + npoints
            x_r = self.trajectory[:, id_s:id_e]
        else:
            # Take a point and propagate the kinematics
            id_s = int(round(t / self.dt))
            x_start = self.trajectory[:, id_s]
            x_r = self.forward_propagate(x_start, npoints)

        return x_r

    def get_initial_pose(self):
        """
        Helper function to get a starting state, depending on the dynamics type.

        :return: starting state
        :rtype: np.ndarray
        """
        x0 = np.zeros((self.n, 1))
        x0[0] = 13.0
        x0[1] = -9.5
        x0[2] = 5.2
        x0[8] = 1.0

        return x0

    def get_static_setpoint(self):
        """
        Helper function to get the initial state of Honey for setpoint stabilization.
        """
        xd = np.array([[11, -7, 4.8, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]]).T
        return xd

    def get_limits(self):
        """
        Get Astrobee control and state limits for ISS

        :return: state and control limits
        :rtype: np.ndarray, np.ndarray
        """
        u_lim = np.array([[0.85, 0.41, 0.41, 0.085, 0.041, 0.041]]).T

        x_lim = np.array([[13, 13, 13,
                           0.5, 0.5, 0.5,
                           1, 1, 1, 1,
                           0.1, 0.1, 0.1]]).T

        return u_lim, x_lim

    # ----------------------------------------
    #               Unit Tests
    # ----------------------------------------

    def test_forward_propagation(self):
        """
        Unit test to check if forward propagation is correctly implemented.
        """

        x0 = np.array([[11, 0.3, 0.4, 0, 0.1, 0, 0, 0, 0, 1, 0.1, 0, 0]]).T
        x_r = self.forward_propagate(x0, 30)
        xd = np.array([11.5, 0.54, 0.4, 0.0, 0.1, 0.0, 0.11971121, 0.0, 0.0, 0.99280876, 0.1, 0.0, 0.0])
        eps = np.linalg.norm(x_r[:, 24] - xd)
        if eps > 1e-2:
            print("Forward propagation has a large error. Double check your dynamics.")
            exit()

    def test_dynamics(self):
        """
        Unit test to check if the Astrobee dynamics are correctly set.
        """
        x0 = np.array([[11, 0.3, 0.4, 0, 0.1, 0, 0, 0, 0, 1, 0.1, 0, 0]]).T
        u0 = np.array([[.1, .1, .1, .01, .01, .01]])
        xd = np.array([[11.0001, 0.310052, 0.400052,
                        0.00104168, 0.101036, 0.00104685,
                        0.00516295, 0.000174902, 0.000154287, 0.999987,
                        0.106519, 0.0070057, 0.00615902]]).T
        xt = self.model(x0, u0)
        eps = np.linalg.norm(np.array(xt) - xd)
        if eps > 1e-4:
            print("Forward propagation has a large error. Double check your dynamics.")
            exit()
