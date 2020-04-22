#!/usr/bin/env python

import numpy as np
import math
from stateEstimation import KF
from scipy.integrate import ode
import sys

# Parameters
sim_time = 10                        # Simulation time
init_states = [math.pi, 0]           # Initial states
sensor_err = 0.05                    # Introduce gaussian error in simulated sensor measurements

# Flags
plot = False        # Plot states and simulated output
anim = True         # Animate pendulum

# We define a pendulum class for convenience, and for using its output function in the ODE solver
class Pendulum:
    def __init__(self, g=9.8156, L=1, l=1, m=1, T=1, **kwargs):
        # Constructive parameters
        self.g = g
        self.L = L
        self.l = l
        self.m = m

        # Pendulum linearized continuous-time state-space (around eq. point th=0)
        self.x = np.zeros((2, 1), dtype=np.float)
        self.A = np.array([[0               ,                1],
                           [-(self.g/self.L), -(self.l/self.m)]], dtype=np.float)
        self.B = np.zeros((2, 1), dtype=np.float)
        self.C = np.array([[1, 0]], dtype=np.float)
        
        self.T = T
        self.u = np.zeros((self.B.shape[1], 1), dtype=np.float)

    # Update non-linear state matrix A
    def update(self, x):
        A = np.array([[0                            ,                1],
                      [-(self.g/self.L)*np.cos(x[0]), -(self.l/self.m)]], dtype=np.float)
        return A
    
    def output(self, t, x, u=0):
        dx = self.A.dot(x.reshape(self.x.shape)) + self.B.dot(u.reshape(self.u.shape))
        return dx

if __name__ == '__main__':
    # Instantiate model (sampling time of 0.005 seconds)
    pend = Pendulum(T=0.005)

    # Define measurement model matrix for Pendulum (2 states, 2 "sensors")
    H = np.array([[1, 0],
                  [0, 1]], dtype=np.float)

    # Define model uncertainty (2 states)
    Q = np.diag([1, 1])

    # Define sensor covariance matrix (1 "sensor")
    R = np.diag([0.1, 0.1])
    
    # Instantiate filter with model
    filt = KF(pend.A, pend.B, H, Q, R, T=pend.T)
    
    # Setup Nonstiff Ordinary Diff. Equation (ODE) solver (equivalent to matlab's ODE45)
    dt = 1e-3       # ODE derivation time
    solv = ode(pend.output).set_integrator('dopri5', method='rtol')   

    # Run for some seconds  
    x = np.zeros((filt.A.shape[0],1))
    u = np.ones((filt.B.shape[1],1))
    t = [0]  # Time vector
    y = np.array(init_states).reshape((len(init_states),1))   # Initial states

    while True:
        # Solve ODE (simulate based on model)
        solv.set_initial_value(y[:,-1])     # Current initial value is last state
        solv.set_f_params(u)                # Apply control input into system
        while solv.successful() and solv.t < filt.T:
            solv.integrate(solv.t+dt)
        y = np.c_[y, solv.y[:]]             # Store simulated output

        # Update states (equivalent to sensing)
        filt.z = np.copy(solv.y[:].reshape(solv.y.shape[0],1))
        filt.z += np.random.normal(scale=sensor_err, size=filt.z.shape)

        # Update jacobian matrix
        filt.A = pend.update(filt.x)

        # Run filter
        filt.run()

        # Store estimated states
        x = np.c_[x, filt.x]

        # Append time
        t.append(t[-1]+filt.T)
        if t[-1] >= sim_time:     # If end of simulation, break loop
            break

    # Plot results
    if plot:
        try:
            import matplotlib.pyplot as plt

            # Plot states
            plt.figure()
            t = np.array(t)
            for k in range(x.shape[0]):
                plt.plot(t, x[k,:], lw=2.0)
            plt.xlabel('Time (s)')
            plt.ylabel('x')
            plt.title('States')
            legend = []
            for k in range(0,x.shape[0]):
                legend.append('x%d' % (k+1))
            plt.legend(legend)
            plt.grid()

            # Plot outputs
            plt.figure()
            for k in range(y.shape[0]):
                plt.plot(t, y[k,:], lw=2.0)
            plt.xlabel('Time (s)')
            plt.ylabel('Angular velocity (rad/s)')
            plt.title('Outputs')
            legend = []
            for k in range(0,y.shape[0]):
                legend.append('y%d' % (k+1))
            plt.legend(legend)
            plt.grid()

            # Show figures
            plt.show()
        except ImportError:
            pass

    # Animate!
    # Based on matplotlib double pendulum example: https://matplotlib.org/3.2.1/gallery/animation/double_pendulum_sgskip.html
    if anim:
        try:
            import matplotlib.pyplot as plt
            import matplotlib.animation as animation
            
            x1 = pend.L*np.sin(x[0, :])
            y1 = -pend.L*np.cos(x[0, :])

            x2 = pend.L*np.sin(y[0, :])
            y2 = -pend.L*np.cos(y[0, :])

            fig = plt.figure()
            ax = fig.add_subplot(111, autoscale_on=False, xlim=(-2.2, 2.2), ylim=(-2, 2))
            ax.grid()
            ax.set_aspect('equal', 'box')

            line1, = ax.plot([], [], 'o-', lw=2)
            line2, = ax.plot([], [], 'o-', lw=2)
            time_template = 'time = %.1fs'
            time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

            ax.legend(['Estimated', 'Real'])

            def init():
                line1.set_data([], [])
                line2.set_data([], [])
                time_text.set_text('')
                return line1, line2, time_text

            def animate(i):
                thisx1 = [0, x1[i]+0]
                thisy1 = [0, y1[i]]
                thisx2 = [0, x2[i]-0]
                thisy2 = [0, y2[i]]

                line1.set_data(thisx1, thisy1)
                line2.set_data(thisx2, thisy2)
                time_text.set_text(time_template % t[i])
                return line1, line2, time_text

            animator = animation.FuncAnimation(fig, animate, np.arange(1, x.shape[1]),
                                        interval=pend.T, blit=True, init_func=init)

            # animator.save('pendulum.mp4', fps=30)
            plt.show()

        except ImportError:
            pass