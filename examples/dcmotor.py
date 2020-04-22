#!/usr/bin/env python

import numpy as np
from stateEstimation import KF
from scipy.integrate import ode
import sys

# Parameters
sim_time = 10                        # Simulation time
init_states = [0, 0]                 # Initial states
sensor_err = 0.2                     # Introduce gaussian error in simulated sensor measurements

# We define a DCMotor class for convenience, and for using its output function in the ODE solver
class DCMotor:
    def __init__(self, Ra=8, La=170e-3, J=10e-3, b=3e-3, If=0.5, kt=0.521, kw=0.521, T=0.001, **kwargs):
        # Constructive parameters
        self.Ra = Ra
        self.La = La
        self.J = J
        self.b = b
        self.If = If
        self.kt = kt
        self.kw = kw

        # Motor continuous-time state-space
        self.A = np.array([[-self.b/self.J,      self.kt*self.If/self.J],
                           [-self.kw*self.If/self.La, -self.Ra/self.La]])
        self.B = np.array([0, 1/self.La]).reshape((2,1))
        self.C = np.array([[1, 0]], dtype=np.float)
        self.dist = np.array([[-1/self.J, 0]]).T         # Input Disturbance

        self.T = T
        self.x = np.zeros((self.A.shape[1],1), dtype=np.float)
        self.u = np.zeros((self.B.shape[1],1), dtype=np.float)
        
    def output(self, t, x, u=0):
        dx = self.A.dot(x.reshape(self.x.shape)) + self.B.dot(u.reshape(self.u.shape)) # + self.dist
        return dx

if __name__ == '__main__':
    # Instantiate DC Motor model (sampling time of 0.05 seconds)
    motor = DCMotor(T=0.005)

    # Define measurement model matrix for DC motor (2 states, 2 "sensors")
    H = np.array([[1, 0],
                  [0, 1]], dtype=np.float)

    # Define model uncertainty for DC motor (2 states)
    Q = np.diag([10, 10])

    # Define sensor covariance matrix for DC motor (1 "sensor")
    R = np.diag([0.01, 0.01])
    
    # Instantiate filter with DC motor model
    filt = KF(motor.A, motor.B, H, Q, R, T=motor.T)
    
    # Setup Nonstiff Ordinary Diff. Equation (ODE) solver (equivalent to matlab's ODE45)
    dt = 1e-3       # ODE derivation time
    solv = ode(motor.output).set_integrator('dopri5', method='rtol')   

    # Run for some seconds
    x = np.zeros((filt.A.shape[0],1))
    u = 10*np.ones((filt.B.shape[1],1))
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

        # Run filter
        filt.run()

        # Store estimated states
        x = np.c_[x, filt.x]

        # Append time
        t.append(t[-1]+filt.T)
        if t[-1] >= sim_time:     # If end of simulation, break loop
            break

    # Plot results
    try:
        import matplotlib.pyplot as plt

        legend = []

        # Plot states
        plt.figure()
        t = np.array(t)
        for k in range(x.shape[0]):
            plt.plot(t, x[k,:], lw=2.0)
        plt.xlabel('Time (s)')
        plt.ylabel('x')
        for k in range(0,x.shape[0]):
            legend.append('Estimated x%d' % (k+1))

        # Plot outputs
        for k in range(y.shape[0]):
            plt.plot(t, y[k,:], lw=2.0)
        plt.xlabel('Time (s)')
        plt.ylabel('Angular velocity (rad/s)')
        for k in range(0,y.shape[0]):
            legend.append('Simulated x%d' % (k+1))

        # Show figures
        plt.legend(legend)
        plt.grid()
        plt.show()
    except ImportError:
        pass