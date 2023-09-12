import numpy as np
import matplotlib.pyplot as plt


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
        self.broke_thruster = False

        # Plotting definitions
        self.plt_window = float("inf")    # running plot window, in seconds, or float("inf")

    def run(self, x0, x_ref=None, online_plot=False):
        """
        Run simulator with specified system dynamics and control function.
        """

        print("Running simulation....")
        sim_loop_length = int(self.total_sim_time / self.dt)  # account for 0th
        t = np.array([0])
        self.sim_3d = False
        if self.model.n == 8:
            self.sim_3d = True
        x_vec = np.array([x0]).reshape(self.model.n, 1)
        u_vec = np.empty((self.model.m, 0))
        if x_ref is None:
            print("Please set reference with 'x_ref' argument.")
            exit()
        e_vec = np.zeros((x_ref.shape[0], 1))

        # Start figure
        if online_plot:
            fig, (ax1, ax2, ax3, ax4) = plt.subplots(4)
        for i in range(sim_loop_length):

            # Get control input and obtain next state
            x = x_vec[:, -1].reshape(self.model.n, 1)
            u = np.array(self.controller[i])
            if self.broke_thruster:
                u[1] = u[1] + np.random.uniform(-0.07, 0.07, (1, 1))
            x_next = self.dynamics(x, u)

            # Store data
            t = np.append(t, t[-1] + self.dt)
            x_vec = np.append(x_vec, np.array(x_next).reshape(self.model.n, 1), axis=1)
            u_vec = np.append(u_vec, u.reshape(self.model.m, 1), axis=1)

            if not self.sim_3d:
                error_p = x_ref[0:2, i].reshape(2, 1) - x[0:2]
                error_att = x_ref[2, i] - x[4]
            else:
                error_p = x_ref[0:3, i].reshape(3, 1) - x[0:3]
                error_att = x_ref[3, i] - x[6]
            error = np.concatenate((error_p, error_att.reshape(1, 1)), axis=0)
            e_vec = np.append(e_vec, np.array(error).reshape(x_ref.shape[0], 1), axis=1)

            if online_plot:
                # Get plot window values:
                if self.plt_window != float("inf"):
                    l_wnd = 0 if int(i + 1 - self.plt_window / self.dt) < 1 else int(i + 1 - self.plt_window / self.dt)
                else:
                    l_wnd = 0

                if not self.sim_3d:
                    ax1.clear()
                    ax1.set_title("Astrobee")
                    ax1.plot(t[l_wnd:], x_vec[0, l_wnd:], 'r--')
                    ax1.plot(t[l_wnd:], x_vec[1, l_wnd:], 'g--')
                    ax1.legend(["x1", "x2"])
                    ax1.set_ylabel("Position [m]")

                    ax2.clear()
                    ax2.plot(t[l_wnd:], x_vec[2, l_wnd:], 'r--')
                    ax2.plot(t[l_wnd:], x_vec[3, l_wnd:], 'g--')
                    ax2.legend(["x3", "x4"])
                    ax2.set_ylabel("Velocity [m/s]")

                    ax3.clear()
                    ax3.plot(t[l_wnd:], x_vec[4, l_wnd:], 'r--')
                    ax3.plot(t[l_wnd:], x_vec[5, l_wnd:], 'g--')
                    ax3.legend(["x5", "x6"])
                    ax3.set_ylabel("Attitude [rad] / Ang. velocity [rad/s]")

                    ax4.clear()
                    ax4.plot(t[l_wnd:-1], u_vec[0, l_wnd:], 'r--')
                    ax4.plot(t[l_wnd:-1], u_vec[1, l_wnd:], 'g--')
                    ax4.plot(t[l_wnd:-1], u_vec[2, l_wnd:], 'b--')
                    ax4.legend(["u1", "u2", "u3"])
                    ax4.set_ylabel("Force input [n] / Torque [nm]")
                    ax4.set_xlabel("Time [s]")
                else:
                    ax1.clear()
                    ax1.set_title("Astrobee")
                    ax1.plot(t[l_wnd:], x_vec[0, l_wnd:], 'r--')
                    ax1.plot(t[l_wnd:], x_vec[1, l_wnd:], 'g--')
                    ax1.plot(t[l_wnd:], x_vec[2, l_wnd:], 'b--')
                    ax1.legend(["x1", "x2", "x3"])
                    ax1.set_ylabel("Position [m]")

                    ax2.clear()
                    ax2.plot(t[l_wnd:], x_vec[3, l_wnd:], 'r--')
                    ax2.plot(t[l_wnd:], x_vec[4, l_wnd:], 'g--')
                    ax2.plot(t[l_wnd:], x_vec[5, l_wnd:], 'b--')
                    ax2.legend(["x4", "x5", "x6"])
                    ax2.set_ylabel("Velocity [m/s]")

                    ax3.clear()
                    ax3.plot(t[l_wnd:], x_vec[6, l_wnd:], 'r--')
                    ax3.plot(t[l_wnd:], x_vec[7, l_wnd:], 'g--')
                    ax3.legend(["x5", "x6"])
                    ax3.set_ylabel("Attitude [rad] / Ang. velocity [rad/s]")

                    ax4.clear()
                    ax4.plot(t[l_wnd:-1], u_vec[0, l_wnd:], 'r--')
                    ax4.plot(t[l_wnd:-1], u_vec[1, l_wnd:], 'g--')
                    ax4.plot(t[l_wnd:-1], u_vec[2, l_wnd:], 'b--')
                    ax4.plot(t[l_wnd:-1], u_vec[3, l_wnd:], 'k--')
                    ax4.legend(["u1", "u2", "u3", "u4"])
                    ax4.set_ylabel("Force input [n] / Torque [nm]")
                    ax4.set_xlabel("Time [s]")

                plt.pause(0.001)

        if online_plot:
            plt.show()

        # Store data internally for offline plotting
        self.t = t
        self.x_vec = x_vec
        self.u_vec = u_vec
        self.e_vec = e_vec
        self.sim_loop_length = sim_loop_length

        return t, x_vec, u_vec

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
        if not self.sim_3d:
            ax1.clear()
            ax1.set_title("Astrobee")
            ax1.plot(t, x_vec[0, :], 'r--')
            ax1.plot(t, x_vec[1, :], 'g--')
            ax1.legend(["x1", "x2"])
            ax1.set_ylabel("Position [m]")

            ax2.clear()
            ax2.plot(t, x_vec[2, :], 'r--')
            ax2.plot(t, x_vec[3, :], 'g--')
            ax2.legend(["x3", "x4"])
            ax2.set_ylabel("Velocity [m/s]")

            ax3.clear()
            ax3.plot(t, x_vec[4, :], 'r--')
            ax3.plot(t, x_vec[5, :], 'g--')
            ax3.legend(["x5", "x6"])
            ax3.set_ylabel("Attitude [rad] / Ang. velocity [rad/s]")

            ax4.clear()
            ax4.plot(t[:-1], u_vec[0, :], 'r--')
            ax4.plot(t[:-1], u_vec[1, :], 'g--')
            ax4.plot(t[:-1], u_vec[2, :], 'b--')
            ax4.legend(["u1", "u2", "u3"])
            ax4.set_ylabel("Force input [n] / Torque [nm]")
            ax4.set_xlabel("Time [s]")
        else:
            ax1.clear()
            ax1.set_title("Astrobee")
            ax1.plot(t, x_vec[0, :], 'r--')
            ax1.plot(t, x_vec[1, :], 'g--')
            ax1.plot(t, x_vec[2, :], 'b--')
            ax1.legend(["x1", "x2", "x3"])
            ax1.set_ylabel("Position [m]")

            ax2.clear()
            ax2.plot(t, x_vec[3, :], 'r--')
            ax2.plot(t, x_vec[4, :], 'g--')
            ax2.plot(t, x_vec[5, :], 'b--')
            ax2.legend(["x4", "x5", "x6"])
            ax2.set_ylabel("Velocity [m/s]")

            ax3.clear()
            ax3.plot(t, x_vec[6, :], 'r--')
            ax3.plot(t, x_vec[7, :], 'g--')
            ax3.legend(["x5", "x6"])
            ax3.set_ylabel("Attitude [rad] / Ang. velocity [rad/s]")

            ax4.clear()
            ax4.plot(t[:-1], u_vec[0, :], 'r--')
            ax4.plot(t[:-1], u_vec[1, :], 'g--')
            ax4.plot(t[:-1], u_vec[2, :], 'b--')
            ax4.plot(t[:-1], u_vec[3, :], 'k--')
            ax4.legend(["u1", "u2", "u3", "u4"])
            ax4.set_ylabel("Force input [n] / Torque [nm]")
            ax4.set_xlabel("Time [s]")

        plt.show()

    def visualize_error(self, x_pred=None):
        """
        Offline plotting of simulation data
        """
        variables = list([self.t, self.e_vec, self.u_vec, self.sim_loop_length])
        if any(elem is None for elem in variables):
            print("Please run the simulation first with the method 'run'.")

        t = self.t
        e_vec = self.e_vec
        u_vec = self.u_vec

        fig, (ax1, ax2, ax3) = plt.subplots(3)
        if not self.sim_3d:
            ax1.clear()
            ax1.set_title("Trajectory Error and Control Effort")
            ax1.plot(t, e_vec[0, :], 'r--')
            ax1.plot(t, e_vec[1, :], 'g--')
            if x_pred is not None:
                x_np = np.asarray(x_pred).reshape(self.x_vec.shape)
                ax1.plot(t, x_np[0, :], 'r')
                ax1.plot(t, x_np[1, :], 'g')
                ax1.legend(["x1", "x2", "x*1", "x*2"])
            else:
                ax1.legend(["x1", "x2"])
            ax1.set_ylabel("Position [m]")

            ax2.clear()
            ax2.plot(t, e_vec[2, :], 'r--')
            ax2.legend(["x5", "x6"])
            ax2.set_ylabel("Attitude [rad]")

            ax3.clear()
            ax3.plot(t[:-1], u_vec[0, :], 'r--')
            ax3.plot(t[:-1], u_vec[1, :], 'g--')
            ax3.plot(t[:-1], u_vec[2, :], 'b--')
            ax3.legend(["u1", "u2", "u3"])
            ax3.set_ylabel("Force input [n] / Torque [nm]")
            ax3.set_xlabel("Time [s]")
        else:
            ax1.clear()
            ax1.set_title("Trajectory Error and Control Effort")
            ax1.plot(t, e_vec[0, :], 'r--')
            ax1.plot(t, e_vec[1, :], 'g--')
            ax1.plot(t, e_vec[2, :], 'b--')
            ax1.legend(["x1", "x2", "x3"])
            ax1.set_ylabel("Position [m]")

            ax2.clear()
            ax2.plot(t, e_vec[3, :], 'r--')
            ax2.legend(["x5", "x6"])
            ax2.set_ylabel("Attitude [rad] ")

            ax3.clear()
            ax3.plot(t[:-1], u_vec[0, :], 'r--')
            ax3.plot(t[:-1], u_vec[1, :], 'g--')
            ax3.plot(t[:-1], u_vec[2, :], 'b--')
            ax3.plot(t[:-1], u_vec[3, :], 'k--')
            ax3.legend(["u1", "u2", "u3", "u4"])
            ax3.set_ylabel("Force input [n] / Torque [nm]")
            ax3.set_xlabel("Time [s]")

        plt.show()

    def visualize_prediction_vs_reference(self, x_pred, x_ref, control):
        fig, (ax1, ax2, ax3) = plt.subplots(3)
        variables = list([self.t])
        if any(elem is None for elem in variables):
            print("Please run the simulation first with the method 'run'.")

        t = self.t
        x_pred = np.asarray(x_pred).T.reshape(6, 301)
        u = np.asarray(control).T.reshape(3, 300)
        ax1.clear()
        ax1.set_title("Prediction vs Reference")
        ax1.plot(t, x_pred[0, :], 'r')
        ax1.plot(t, x_pred[1, :], 'g')
        ax1.plot(t[:-1], x_ref[0, :], 'r--')
        ax1.plot(t[:-1], x_ref[1, :], 'g--')
        ax1.set_ylabel("Position [m]")

        ax2.clear()
        ax2.plot(t, x_pred[4, :], 'r')
        ax2.plot(t[:-1], x_ref[2, :], 'r--')
        ax2.legend(["x*5", "x6"])
        ax2.set_ylabel("Attitude [rad] ")

        ax3.clear()
        ax3.plot(t[:-1], u[0, :], 'r')
        ax3.plot(t[:-1], u[1, :], 'g')
        ax3.plot(t[:-1], u[2, :], 'b')
        ax3.legend(["Fx", "Fy", "Fz"])
        ax3.set_ylabel("Control")
        ax3.set_xlabel("Time [s]")
        plt.show()
        return

    def visualize_state_vs_reference(self, state, ref, control):
        fig, (ax1, ax2, ax3) = plt.subplots(3)
        variables = list([self.t])
        if any(elem is None for elem in variables):
            print("Please run the simulation first with the method 'run'.")

        t = self.t
        u = np.asarray(control).T.reshape(3, 300)
        ax1.clear()
        ax1.set_title("State vs Reference")
        ax1.plot(t, state[0, :], 'r')
        ax1.plot(t, state[1, :], 'g')
        ax1.plot(t[:-1], ref[0, :], 'r--')
        ax1.plot(t[:-1], ref[1, :], 'g--')

        ax1.set_ylabel("Position [m]")

        ax2.clear()
        ax2.plot(t, state[4, :], 'r')
        ax2.plot(t[:-1], ref[2, :], 'r--')
        ax2.legend(["x5", "x6"])
        ax2.set_ylabel("Attitude [rad] ")

        ax3.clear()
        ax3.plot(t[:-1], u[0, :], 'r')
        ax3.plot(t[:-1], u[1, :], 'g')
        ax3.plot(t[:-1], u[2, :], 'b')
        ax3.legend(["Fx", "Fy", "Fz"])
        ax3.set_ylabel("Control")
        ax3.set_xlabel("Time [s]")
        plt.show()
        return

    def set_window(self, window):
        """
        Set the plot window length, in seconds.

        :param window: window length [s]
        :type window: float
        """
        self.plt_window = window

    def broken_thruster(self, act=True):
        """
        Simulate Astrobee broken thruster.

        :param act: activate or deactivate broken thruster
        :type act: boolean
        """

        self.broke_thruster = True
