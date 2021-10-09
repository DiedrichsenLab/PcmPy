"""
Pattern Component Modelling (PCM) toolbox for Python
----------------------------------------------------

Documentation is available in the docstrings and online at
https://pcm-toolbox-python.readthedocs.io
"""
from .sim import *
from .dataset import *
from .inference import *
from .model import *
from .optimize import *
from .regression import *
from .util import *
from .vis import *
from .matrix import *

__all__ = ['sim', 'matrix', 'inference', 'optimize',
           'model', 'util', 'vis', 'regression']
