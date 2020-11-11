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

def r2_score(Y,Yp):
    """
         R2 score without subtracting intercept
    """
    return 1-np.sum((Y-Yp)**2)/np.sum(Y**2)

def method_compare(comp, theta, num_sim = 10, N = 50, P = 100,
        alpha_spacing = exp(np.linspace(-1,5,10)),fit_intercept = False):
    Q = comp.shape[0]
    s = sqrt(exp(theta[comp]))
    z = np.zeros((num_sim,))

    # Prepare output structure
    n_param = theta.shape[0]
    df = {}
    for i in range(n_param):
        df.update({'theta' + str(i): z})
    df.update({'R2_Pcm':z,'R2_Lin':z,'R2_Rid':z,'alpha':z,'time_Pcm':z,'time_Rid':z})
    T = pd.DataFrame(df)

    # Prepare Ridge regression
    M = pcm.regression.RidgeDiag(comp, fit_intercept = fit_intercept)
    R = Ridge(alpha=30.0, fit_intercept=fit_intercept)
    # set the tolerance to a large value to make the example faster
    R_pipe = Pipeline(steps=[('ridge', R)])
    param_grid = {
        'ridge__alpha': alpha_spacing
        }

    # Make scorer
    r2_scorer = make_scorer(r2_score, greater_is_better=True)

    L = LinearRegression(fit_intercept=False)

    for i in range(num_sim):
        # Make training data
        U = np.random.normal(0,1,(Q,P))
        U = U * s.reshape((Q,1))
        Z = np.random.normal(0,1,(N,Q))
        Y = Z @ U + np.random.normal(0,sqrt(exp(theta[-1])),(N,P))

        # Make test data
        Zt = np.random.normal(0,1,(N,Q))
        Yt = Zt @ U + np.random.normal(0,1,(N,P))

        # PCM regression
        t0 = time.perf_counter()
        M.optimize_regularization(Z,Y)
        M.fit(Z,Y)
        Yp = M.predict(Zt)

        T['time_Pcm'][i] = time.perf_counter() - t0
        for j in range(n_param):
            T['theta'+str(j)][i]=M.theta_[j]
        T['R2_Pcm'][i] = r2_score(Yt,Yp)

        # Linear regression
        L.fit(Z,Y)
        Yp2 = L.predict(Zt)
        T['R2_Lin'][i] = r2_score(Yt,Yp2)

        # Ridge regression
        t0 = time.perf_counter()
        R_search = GridSearchCV(R_pipe, param_grid, n_jobs=-1, scoring = r2_scorer)
        R_search.fit(Z,Y)
        R_pipe.set_params(**R_search.best_params_)
        R_pipe.fit(Z,Y)
        Yp3 = R_pipe.predict(Zt)
        T['time_Rid'][i] = time.perf_counter() - t0
        T['alpha'][i]=R_search.best_params_['ridge__alpha']
        T['R2_Rid'][i] = r2_score(Yt,Yp3)
    return T

def pcm_performance(comp, theta, num_sim = 10, N = 50, P = 100,
        alpha_spacing = exp(np.linspace(-1,5,10)),fixed_effect = None, fit_intercept = False):
    Q = comp.shape[0]
    s = sqrt(exp(theta[comp]))
    z = np.zeros((num_sim,))

    # Prepare output structure
    n_param = theta.shape[0]
    df = {}
    for i in range(n_param):
        df.update({'theta' + str(i): z})
    df.update({'R2_Pcm':z,'time_Pcm':z})
    T = pd.DataFrame(df)

    # Prepare Ridge regression
    M = pcm.regression.RidgeDiag(comp, fit_intercept = fit_intercept)

    # Make scorer
    r2_scorer = make_scorer(r2_score, greater_is_better=True)

    for i in range(num_sim):
        # Make training data
        U = np.random.normal(0,1,(Q,P))
        U = U * s.reshape((Q,1))
        Z = np.random.normal(0,1,(N,Q))
        Y = Z @ U + np.random.normal(0,sqrt(exp(theta[-1])),(N,P))

        # Make test data
        Zt = np.random.normal(0,1,(N,Q))
        Yt = Zt @ U + np.random.normal(0,1,(N,P))

        # PCM regression
        t0 = time.perf_counter()
        M.optimize_regularization(Z,Y)
        M.fit(Z,Y)
        Yp = M.predict(Zt)

        T['time_Pcm'][i] = time.perf_counter() - t0
        for j in range(n_param):
            T['theta'+str(j)][i]=M.theta_[j]
        T['R2_Pcm'][i] = r2_score(Yt,Yp)
    return T
