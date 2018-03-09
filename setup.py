"""
Setup of Traffic Camera Streaming Pipeline
Author: Jim Ren
"""
from setuptools import setup

setup(name='tcp',
      version='0.1.dev0',
      description='Pipeline to extract trajectories from traffic camera streams',
      author='Jim Ren',
      author_email='jim.x.ren@berkeley.edu',
      package_dir = {'': 'src'},
      packages=['tcp', 'tcp.object_detection', 'tcp.streaming', 'tcp.registration', 'tcp.configs', 'tcp.utils'],
     )
