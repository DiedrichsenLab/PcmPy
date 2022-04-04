"""
Pattern Component Modelling (PCM) toolbox for Python
----------------------------------------------------

Documentation is available in the docstrings and online at
https://pcm-toolbox-python.readthedocs.io
"""
from .dataset import *
from .matrix import *
from .util import *
from .sim import *
from .model import *
from .optimize import *
from .inference import *
from .regression import *
from .vis import *

__all__ = ['sim', 'matrix', 'inference', 'optimize',
           'model', 'util', 'vis', 'regression']
