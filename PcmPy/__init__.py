"""
Pattern Component Modelling (PCM) toolbox for Python 
----------------------------------------------------

Documentation is available in the docstrings and online at
https://pcm-toolbox-python.readthedocs.io
"""
from PcmPy.model import Model
from PcmPy.model import FeatureModel
from PcmPy.model import ComponentModel
from PcmPy.model import FreeModel
from PcmPy.model import CorrelationModel

from PcmPy import sim
from PcmPy import dataset
from PcmPy import inference
from PcmPy import model 
from PcmPy import optimize 
from PcmPy import regression
from PcmPy import util
from PcmPy import vis

__all__ = ['sim', 'matrix', 'inference', 'optimize',
           'model', 'util', 'vis', 'regression']
