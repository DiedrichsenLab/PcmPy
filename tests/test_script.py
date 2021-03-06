#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_indicator

@author: jdiedrichsen
"""
import sys
print(sys.path)
sys.path.append('/Users/jdiedrichsen/Python')
import unittest
# Import necessary libraries
import PcmPy as pcm
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from numpy import exp, sqrt
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
import time
from test_regression_performance import r2_score, method_compare, pcm_performance


# Make model
TL = pd.DataFrame()
comp  = np.array([0,0,0,0,0,0,0,0]) # 20 regressors in 4 groups
Q = comp.shape[0]
theta = np.array([0.0,1.0]) # Theta's for simulation 
P = 1
N = 1
D = pcm_performance(comp, theta, P = P, N = N)
