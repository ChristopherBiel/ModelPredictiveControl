import numpy as np
import casadi as ca
import matplotlib.pyplot as plt
import time


class EmbeddedSimEnvironment(object):

    def __init__(self, model, dynamics, controller, time=100.0):
        """
        Embedded simulation environment. Simulates the syste given dynamics
        and a control law, plots in matplotlib.

        :param model: model object
        :type model: object
        :param dynamics: system dynamics function (x, u)
        :type dynamics: casadi.DM
        :param controller: controller function (x, r)
        :type controller: casadi.DM
        :param time: total simulation time, defaults to 100 seconds
        :type time: float, optional
        """
        self.model = model
        self.dynamics = dynamics
        self.controller = controller
        self.total_sim_time = time  # seconds
        self.dt = self.model.dt
        self.estimation_in_the_loop = False

    def run(self, x0):
        """
        Run simulator with specified system dynamics and control function.
        """

        # print("Running simulation....")
        sim_loop_length = int(self.total_sim_time / self.dt) + 1  # account for 0th
        t = np.array([0])
        st = np.array([0])
        x_vec = np.array([x0]).reshape(self.model.n, 1)
        u_vec = np.empty((6, 0))
        e_vec = np.empty((12, 0))

        for i in range(sim_loop_length):

            # Get control input and obtain next state
            x = x_vec[:, -1].reshape(self.model.n, 1)
            u, error, solve_time = self.controller(x, i * self.dt)
            x_next = self.dynamics(x, u)
            x_next[6:10] = x_next[6:10] / ca.norm_2(x_next[6:10])

            # Store data
            t = np.append(t, t[-1] + self.dt)
            st = np.append(st, solve_time)
            x_vec = np.append(x_vec, np.array(x_next).reshape(self.model.n, 1), axis=1)
            u_vec = np.append(u_vec, np.array(u).reshape(self.model.m, 1), axis=1)
            e_vec = np.append(e_vec, error.reshape(12, 1), axis=1)

        _, error, solve_time = self.controller(x_next, i * self.dt)
        st = np.append(st, solve_time)
        e_vec = np.append(e_vec, error.reshape(12, 1), axis=1)

        self.t = t
        self.st = st
        self.x_vec = x_vec
        self.u_vec = u_vec
        self.e_vec = e_vec
        self.sim_loop_length = sim_loop_length
        return t, st, x_vec, u_vec, e_vec

    def visualize(self):
        """
        Offline plotting of simulation data
        """
        variables = list([self.t, self.x_vec, self.u_vec, self.sim_loop_length])
        if any(elem is None for elem in variables):
            print("Please run the simulation first with the method 'run'.")

        t = self.t
        x_vec = self.x_vec
        u_vec = self.u_vec

        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4)
        fig2, (ax5, ax6) = plt.subplots(2)
        ax1.clear()
        ax1.set_title("Astrobee States")
        ax1.plot(t, x_vec[0, :], 'r--',
                 t, x_vec[1, :], 'g--',
                 t, x_vec[2, :], 'b--')
        ax1.legend(["x1", "x2", "x3"])
        ax1.set_ylabel("Position [m]")
        ax1.grid()

        ax2.clear()
        ax2.plot(t, x_vec[3, :], 'r--',
                 t, x_vec[4, :], 'g--',
                 t, x_vec[5, :], 'b--')
        ax2.legend(["x3", "x4", "x5"])
        ax2.set_ylabel("Velocity [m/s]")
        ax2.grid()

        ax3.clear()
        ax3.plot(t, x_vec[6, :], 'r--',
                 t, x_vec[7, :], 'g--',
                 t, x_vec[8, :], 'b--')
        ax3.legend(["x6", "x7", "x8"])
        ax3.set_ylabel("Attitude [rad]")
        ax3.grid()

        ax4.clear()
        ax4.plot(t, x_vec[10, :], 'r--',
                 t, x_vec[11, :], 'g--',
                 t, x_vec[12, :], 'b--')
        ax4.legend(["x9", "x10", "x11"])
        ax4.set_ylabel("Ang. velocity [rad/s]")
        ax4.grid()

        # Plot control input
        ax5.clear()
        ax5.set_title("Astrobee Control inputs")
        ax5.plot(t[:-1], u_vec[0, :], 'r--',
                 t[:-1], u_vec[1, :], 'g--',
                 t[:-1], u_vec[2, :], 'b--')
        ax5.legend(["u0", "u1", "u2"])
        ax5.set_ylabel("Force input [N]")
        ax5.grid()

        ax6.clear()
        ax6.plot(t[:-1], u_vec[3, :], 'r--',
                 t[:-1], u_vec[4, :], 'g--',
                 t[:-1], u_vec[5, :], 'b--')
        ax6.legend(["u3", "u4", "u5"])
        ax6.set_ylabel("Torque input [Nm]")
        ax6.grid()

        plt.show()

    def visualize_error(self):
        """
        Offline plotting of simulation data
        """
        variables = list([self.t, self.e_vec, self.u_vec, self.sim_loop_length])
        if any(elem is None for elem in variables):
            print("Please run the simulation first with the method 'run'.")

        t = self.t
        x_vec = self.e_vec
        u_vec = self.u_vec

        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4)
        fig2, (ax5, ax6) = plt.subplots(2)
        ax1.clear()
        ax1.set_title("Trajectory Error")
        ax1.plot(t, x_vec[0, :], 'r--',
                 t, x_vec[1, :], 'g--',
                 t, x_vec[2, :], 'b--')
        ax1.legend(["x1", "x2", "x3"])
        ax1.set_ylabel("Position Error [m]")
        ax1.grid()

        ax2.clear()
        ax2.plot(t, x_vec[3, :], 'r--',
                 t, x_vec[4, :], 'g--',
                 t, x_vec[5, :], 'b--')
        ax2.legend(["x3", "x4", "x5"])
        ax2.set_ylabel("Velocity Error [m/s]")
        ax2.grid()

        ax3.clear()
        ax3.plot(t, x_vec[6, :], 'r--',
                 t, x_vec[7, :], 'g--',
                 t, x_vec[8, :], 'b--')
        ax3.legend(["ex", "ey", "ez"])
        ax3.set_ylabel("Attitude Error [rad]")
        ax3.grid()

        ax4.clear()
        ax4.plot(t, x_vec[9, :], 'r--',
                 t, x_vec[10, :], 'g--',
                 t, x_vec[11, :], 'b--')
        ax4.legend(["x9", "x10", "x11"])
        ax4.set_ylabel("Ang. velocity Error [rad/s]")
        ax4.grid()

        # Plot control input
        ax5.clear()
        ax5.set_title("Astrobee Control inputs")
        ax5.plot(t[:-1], u_vec[0, :], 'r--',
                 t[:-1], u_vec[1, :], 'g--',
                 t[:-1], u_vec[2, :], 'b--')
        ax5.legend(["u0", "u1", "u2"])
        ax5.set_ylabel("Force input [N]")
        ax5.grid()

        ax6.clear()
        ax6.plot(t[:-1], u_vec[3, :], 'r--',
                 t[:-1], u_vec[4, :], 'g--',
                 t[:-1], u_vec[5, :], 'b--')
        ax6.legend(["u3", "u4", "u5"])
        ax6.set_ylabel("Torque input [Nm]")
        ax6.grid()

        plt.show()

    def calcScore(self, verbosity=True):
        """
        Calculate a performance score for a simulation.
        More is better!
        """
        score = []

        st = self.st
        t  = self.t
        e = self.e_vec


        # Maximum computational time
        max_ct = np.max(st)
        score.append(-1*max(round((max_ct - 0.1) * 100, 3), 0.0) * 0.1)

        # Average score
        avg_ct = np.average(st)
        if avg_ct > 0.1:
            score.append((0.1 - avg_ct) * 30)
        else:
            score.append(max((0.1 - avg_ct), 0.0) * 5)

        # Convergence time
        error = np.zeros((2,e.shape[1]))
        for i in range(e.shape[1]):
            error[:,i] = np.array([np.linalg.norm(e[0:3,i]), np.rad2deg(np.linalg.norm(e[6:9,i]))])
        cvg_v1 = np.where(error[0,:] < 0.05)[0]
        cvg_v2 = np.where(error[1,:] < 10)[0]
        if cvg_v1.size != 0 and cvg_v2.size != 0:
            cvg_i1 = cvg_v1[0]
            cvg_i2 = cvg_v2[0]
            cvg_i = max(cvg_i1, cvg_i2)
            cvg_t = t[cvg_i]
        else: 
            cvg_t = t[-1]
        score.append(max((35.0 - cvg_t), 0.0) * 0.1)

        # Factor in steady-state errors
        ss_p = np.mean(error[0,cvg_i:-1])
        ss_a = np.mean(error[1,cvg_i:-1])
        score.append((0.05 - ss_p) * 100)
        score.append((10 - ss_a) * 1)
        if verbosity:
            print("--------------------------- EVALUATION ---------------------------")
            print("MaxCpT Penalty:                       %5.4fs ->    [  %5.4f   ]" % (max_ct, score[0]))
            print("AvgCpT Score:                         %5.4fs -> [%5.4f /  0.500]" % (avg_ct, score[1]))
            print("ConvT Score:   PosConv %5.2fs, AttConv %5.2fs -> [%5.4f /  3.500]" % (t[cvg_i1], t[cvg_i2], score[2]))
            print("SdySt Pos Score:                 avg. %5.4fm -> [%5.4f /  5.000]" % (ss_p, score[3]))
            print("SdySt Att Score:               avg. %4.4fdeg -> [%5.4f / 10.000]" % (ss_a, score[4]))
            print(f"--------------------- Overall score: %.4f ---------------------" % (sum(score)))

        return sum(score)