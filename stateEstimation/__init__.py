import numpy as np
from scipy.linalg import expm

# Discrete Kalman Filter
# Constructor parameters:
#     - A, B: State-space matrices
#     - H: Measurement model matrix
#     - Q: Model uncertainty matri
#     - R: Sensor covariance matrix
#     - T: Sampling time
#     - discretize: Flag for converting system to discrete-time
class KF:
    def __init__(self, A, B, H, Q, R, T=1, discretize=True, **kwargs):
        #### System State-Space ####
        self.A = A
        self.B = B
        self.x = np.zeros((self.A.shape[0],1), dtype=np.float)
        self.u = np.zeros((self.B.shape[1],1), dtype=np.float)
        self.T = T

        #### Discretization ####
        if discretize:
            self.A = expm(self.A*self.T)
            self.B = np.linalg.inv(A).dot((self.A - np.eye(self.A.shape[0]))).dot(self.B)
        
        #### Kalman Filter ####
        self.H = H                                              # Measurement model matrix
        self.Q = Q                                              # Model uncertainty matrix
        self.R = R                                              # Sensor covariance matrix
        self.P = np.eye(self.A.shape[0], dtype=np.float)        # State covariance matrix
        self.z = np.zeros((self.H.shape[0], 1), dtype=np.float) # Sensor measurement array

    # Prediction step
    # Parameters:
    #   - u: System input
    def predict(self):
        # State-transition estimates (based on system model)
        f = self.A.dot(self.x) + self.B.dot(self.u)
        
        # Next-states and covariance prediction
        self.x += self.T*f
        self.P = self.A.dot(self.P).dot(self.A.T) + self.Q

    # Update step
    # Parameters:
    #   - z: Sensor measurements array
    def update(self):
        S = self.H.dot(self.P).dot(self.H.T) + self.R
        K = self.P.dot(self.H.T).dot(np.linalg.inv(S))
        h = self.H.dot(self.x)
        self.x += K.dot(self.z-h)
        self.P = (np.eye(self.x.shape[0])-K.dot(self.H)).dot(self.P)

    # Run filter
    # Parameters:
    #   - u: System input
    #   - z: Sensor measurements array
    def run(self):
        self.predict()
        self.update()