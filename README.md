# Python package for probabilistic state estimation (so far Kalman Filter only)

This python package implements algorithms for probabilistic state estimation (e.g., Kalman Filter, Particle Filter).
So far it only includes a Kalman Filter implementation module.
Tested and working in Linux Ubuntu 18.06.

## Dependencies

* numpy
* scipy

## Installation

To install the package, you may use [PIP](https://pypi.org/project/pip/). First, clone the repository to your computer:

```
git clone http://github.com/rgmaidana/stateEstimation $HOME/stateEstimation
```

Go to the cloned directory and run the pip local install:

```
pip install -e .
```

Alternatively, you can install the package directly from PyPI's repositories:

```
pip install -U stateEstimation
```

## Usage

To use the package, simply import it in python:

```
import stateEstimation
```

### Kalman Filter

The Kalman Filter module implements the linear Kalman Filter algorithm.
To use it, the user must instantiate the module with the state-space matrices (i.e., A and B), as well as the measurement model matrix (H), 
the model uncertainty matrix (Q), and the sensor covariance matrix (R).

Then the user must call the _run()_ function to 
perform the prediction and update steps. For example:

```
from stateEstimation import KF

# Define the state-space matrices, i.e. A, B

# Define the Kalman Filter matrices, i.e. H, Q, R

# Instantiate Kalman Filter
filter = KF(A, B, H, Q, R)

filter.run()
```

An example of the state estimation of a simulated DC motor can be found in the [examples](https://github.com/rgmaidana/stateEstimation/blob/master/examples/dcmotor.py) folder.
The module may also be used as an Extended Kalman Filter (EKF), an Unscented Kalman Filter (UKF), etc, as the state-space and filter matrices are user-defined 
(i.e., the state-space and measurement model matrices may be linearized by the user).
An example of the filter being used as an EKF to estimate the angle and velocity of a simple pendulum can be found in the [examples](https://github.com/rgmaidana/stateEstimation/blob/master/examples/pendulum.py).

In both examples, normal gaussian noise is used to demonstrate the effects of sensor measurement variance in the state estimation.
The error magnitude can be adjusted in the example scripts (e.g., 0 for no error).

## Collaborators

* [Renan Maidana](https://github.com/rgmaidana)
