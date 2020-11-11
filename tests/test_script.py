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
T = pd.DataFrame()
theta = np.array([-1.0,1.0])
for P in [100]:
    for Q in [10]:
        for N in [1000]:
            comp = np.zeros((Q,1),dtype = int)
            D = method_compare(comp, theta, P = P, N = N)
            D['N']=N
            D['P']=P
            D['Q']=Q
            T.append(D, ignore_index=True)
pass