"""
Model Predictive Control - CasADi interface
Adapted from Helge-André Langåker work on GP-MPC
Customized by Pedro Roque for EL2700 Model Predictive Countrol Course at KTH
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import numpy as np
import casadi as ca
import casadi.tools as ctools


class FiniteOptimization(object):

    def __init__(self, model, dynamics,
                 total_time=10.0, rendezvous_time=5.0,
                 R=None, u_lim=None,
                 ref_type="2d", solver_opts=None):
        """
        Finize optimization solver for minimum energy transfer.

        :param model: system class
        :type model: Python class
        :param dynamics: system dynamics function
        :type dynamics: np.ndarray, ca.DM, ca.MX
        :param total_time: total optimization time, defaults to 10
        :type total_time: float, optional
        :param rendezvous_time: time to rendezvous, defaults to 5
        :type rendezvous_time: float, optional
        :param R: weight matrix for the cost function, defaults to None
        :type R: np.ndarray, optional
        :param ref_type: reference type (2d or 3d - with Z), defaults to '2d'
        :type ref_type: string, optional
        :param solver_opts: optional solver parameters, defaults to None
        :type solver_opts: dictionary, optional
        """

        build_solver_time = -time.time()
        self.dt = model.dt
        self.model = model
        self.Nx, self.Nu = model.n, model.m
        self.Nt = int(total_time / self.dt)
        self.Ntr = int(rendezvous_time / self.dt)
        self.dynamics = dynamics

        if ref_type == "2d":
            self.Nr = 3

            # Rendezvous Tolerances
            self.pos_tol = 0.001 * np.ones((2, 1))
            self.att_tol = 0.001
        else:
            self.Nr = 4

            # Rendezvous Tolerances for 3D
            self.pos_tol = 0.001 * np.ones((3, 1))
            self.att_tol = 0.001

        if u_lim is not None:
            u_lb = -u_lim
            u_ub = u_lim

        # Initialize variables
        self.set_cost_functions()
        self.x_sp = None

        # Cost function weights
        if R is None:
            R = np.eye(self.Nu) * 0.01
        self.R = ca.MX(R)

        # Starting state parameters
        x0 = ca.MX.sym('x0', self.Nx)
        u0 = ca.MX.sym('u0', self.Nu)
        x_t_ref = ca.MX.sym('p_t_ref', self.Nr * (self.Nt - self.Ntr))
        param_s = ca.vertcat(x0, x_t_ref, u0)

        # Create optimization variables structure
        opt_var = ctools.struct_symMX([(ctools.entry('u', shape=(self.Nu,), repeat=self.Nt),
                                        ctools.entry('x', shape=(self.Nx,), repeat=self.Nt + 1),
                                        )])
        self.opt_var = opt_var
        self.num_var = opt_var.size

        # Decision variable boundries
        self.optvar_lb = opt_var(-np.inf)
        self.optvar_ub = opt_var(np.inf)

        # Set initial values
        obj = ca.MX(0)
        con_eq = []
        con_ineq = []
        con_ineq_lb = []
        con_ineq_ub = []
        con_eq.append(opt_var['x', 0] - x0)

        # Generate MPC Problem
        r_i = 0
        for t in range(self.Nt):

            # Get variables
            x_t = opt_var['x', t]
            u_t = opt_var['u', t]

            # Dynamics constraint
            x_t_next = self.dynamics(x_t, u_t)
            con_eq.append(x_t_next - opt_var['x', t + 1])

            # State constraints for rendezvous
            if t > self.Ntr:
                # Make sure that we target the reference then
                if ref_type == "2d":
                    x_ref = x_t_ref[(r_i * self.Nr):(r_i * self.Nr + self.Nr)]

                    # TODO: use 'x_ref', 'x_t', 'self.pos_tol' and 'self.att_tol'
                    #       to define the maximum error tolerance for the position
                    #       in X, Y and angle theta, by adjusting 'con_ineq',
                    #       'con_ineq_ub' and 'con_ineq_lb' - take inspiration from
                    #       the example below for 3D
                else:
                    x_ref = x_t_ref[(r_i * self.Nr):(r_i * self.Nr + self.Nr)]

                    con_ineq.append(x_ref[0:3] - x_t[0:3])
                    con_ineq_ub.append(self.pos_tol)
                    con_ineq_lb.append(-self.pos_tol)

                    con_ineq.append(x_ref[3] - x_t[6])
                    con_ineq_ub.append(self.att_tol)
                    con_ineq_lb.append(-self.att_tol)

                r_i += 1

            # Input constraints
            if u_lim is not None:
                con_ineq.append(u_t)
                con_ineq_ub.append(u_ub)
                con_ineq_lb.append(u_lb)

            # Objective Function / Cost Function
            obj += self.cost_function(u_t, self.R)

        # Equality constraints bounds are 0 (they are equality constraints),
        # -> Refer to CasADi documentation
        num_eq_con = ca.vertcat(*con_eq).size1()
        num_ineq_con = ca.vertcat(*con_ineq).size1()
        con_eq_lb = np.zeros((num_eq_con,))
        con_eq_ub = np.zeros((num_eq_con,))

        # Set constraints
        con = ca.vertcat(*con_eq, *con_ineq)
        self.con_lb = ca.vertcat(con_eq_lb, *con_ineq_lb)
        self.con_ub = ca.vertcat(con_eq_ub, *con_ineq_ub)

        # Build NLP Solver (can also solve QP)
        qp = dict(x=opt_var, f=obj, g=con, p=param_s)
        options = {}
        if solver_opts is not None:
            options.update(solver_opts)
        self.solver = ca.qpsol('qp_solver', 'qrqp', qp, options)

        build_solver_time += time.time()
        print('----------------------------------------')
        print('# Time to build solver: %f sec' % build_solver_time)
        print('# Number of variables: %d' % self.num_var)
        print('# Number of equality constraints: %d' % num_eq_con)
        print('# Number of inequality constraints: %d' % num_ineq_con)
        print('----------------------------------------')
        pass

    def set_cost_functions(self):
        """
        Helper method to create CasADi functions for the MPC cost objective.
        """
        # Create functions and function variables for calculating the cost
        R = ca.MX.sym('R', self.Nu, self.Nu)
        u = ca.MX.sym('u', self.Nu)

        # Instantiate function
        self.cost_function = ca.Function('J', [u, R], [u.T @ R @ u])

    def solve_problem(self, x0, xr):
        """
        Solve the optimization problem.

        :param x0: starting state
        :type x0: np.ndarray
        :param x0: target set of states
        :type x0: np.ndarray
        :return: optimal states and control inputs
        :rtype: np.ndarray
        """

        # Initial state
        u0 = np.zeros(self.Nu)

        # Initialize variables
        self.optvar_x0 = np.full((1, self.Nx), x0.T)

        # Initial guess of the warm start variables
        self.optvar_init = self.opt_var(0)
        self.optvar_init['x', 0] = self.optvar_x0[0]

        print('\nSolving a total of %d time-steps' % self.Nt)
        solve_time = -time.time()

        param = ca.vertcat(x0, xr.ravel(order="F"), u0)
        args = dict(x0=self.optvar_init,
                    lbx=self.optvar_lb,
                    ubx=self.optvar_ub,
                    lbg=self.con_lb,
                    ubg=self.con_ub,
                    p=param)

        # Solve NLP
        sol = self.solver(**args)
        status = self.solver.stats()['return_status']
        optvar = self.opt_var(sol['x'])

        solve_time += time.time()
        print('Solver took %f seconds to obtain a solution.' % (solve_time))
        print('Final cost: ', sol['f'])
        print('Solver status: ', status)

        return optvar['x'], optvar['u']
