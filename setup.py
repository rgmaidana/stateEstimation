from setuptools import setup

setup(name='stateEstimation',
      version='0.1',
      description='Python package for probabilistic state estimation techniques (e.g., KF, EKF, PF)',
      url='http://www.github.com/rgmaidana/stateEstimation',
      author='Renan Maidana',
      author_email='renan.g.maidana@ntnu.no',
      license='MIT',
      packages=['stateEstimation'],
      install_requires=[
          'numpy',
          'scipy'
      ],
      zip_safe=False
    )