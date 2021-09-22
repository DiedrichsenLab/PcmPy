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

__all__ = ['sim', 'matrix', 'inference', 'optimize',
           'model', 'util', 'vis', 'regression']
