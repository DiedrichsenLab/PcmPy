#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_indicator

@author: jdiedrichsen
"""
# Import necessary libraries
import PcmPy as pcm
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from numpy import exp, sqrt
from sklearn.base import BaseEstimator, TransformerMixin
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

class RidgeScaler(BaseEstimator,TransformerMixin):
    def __init__(self, components, theta=None):
        self.components = components.astype(int)
        self.num_comp = np.max(components)+1
        if theta is None:
            theta = np.zeros((self.num_comp,))
        self.theta = theta

    def fit(self,X,y=None):
        return self

    def transform(self,X,y=None):
        X = X*exp(self.theta[self.components])
        return X

    def get_params(self, deep=True):
        return {"components": self.components, "theta": self.theta}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def get_param_grid(self,amin,amax,steps):
        theta_spacing = exp(np.linspace(amin,amax,steps)).reshape(-1,1)
        for i in range(self.num_comp):
            if i==0:
                T = theta_spacing
            else:
                on2 = np.ones((T.shape[0],1))
                on = np.ones((steps,1))
                T = np.c_[np.kron(T,on), np.kron(on2,theta_spacing)]
        return T


def numcomp_compare(num_comp, Qp, theta=None, num_sim = 10, N = 50, P = 100,methods = ['pcm1','pcm2','ridge'],fit_intercept = False):
    z = np.zeros((num_sim * len(methods),))

    # Prepare output structure
    n_param = num_comp+1
    df = {}
    for i in range(n_param):
        df.update({'theta' + str(i): z})
    df.update({'R2':z,'alpha':z,'time':z})
    T = pd.DataFrame(df)

    if (theta is None):
        theta=np.ones((n_params,1))*-2
        theta[-1]=0

    Q = Qp * num_comp
    comp = np.kron(np.array(range(num_comp),dtype=int),np.ones((Qp,),dtype=int))
    s = sqrt(exp(theta[comp]))
    for i in range(num_sim):
        # Make training data
        U = np.random.normal(0,1,(Q,P))
        U = U * s.reshape((Q,1))
        Z = np.random.normal(0,1,(N,Q))
        Y = Z @ U + np.random.normal(0,sqrt(exp(theta[-1])),(N,P))

        # Make test data
        Zt = np.random.normal(0,1,(N,Q))
        Yt = Zt @ U + np.random.normal(0,sqrt(exp(theta[-1])),(N,P))

        for m in methods:
            t0 = time.perf_counter()
            if m=='pcm1'
                M=pcm.regression.RidgeDiagonal(comp)
                M.optimize_regularization(Z,Y)
            elif m=='pcm2'
            elif m=='ridge'
                S  = RidgeScaler(comp)
                R = Ridge(alpha=1.0, fit_intercept=fit_intercept)
                M = Pipeline(steps=[('scaler',S),('ridge', R)])
                r2_scorer = make_scorer(r2_score, greater_is_better=True)
                T = S.get_param_grid(-3,3,7)
                param_grid = {'scaler__theta':T}
                M = GridSearchCV(M, param_grid, n_jobs=-1, scoring = r2_scorer)
            M.fit(Z,Y)
            Yp = M.predict(Zt)
            T['time'][i] = time.perf_counter() - t0
            for j in range(n_param):
                T['theta'+str(j)][i]=M.theta_[j]
            T['R2'][i] = r2_score(Yt,Yp)

    return T

def method_compare(comp, theta, num_sim = 10, N = 50, P = 100,fit_intercept = False):
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

def pcm_performance(comp, theta, num_sim = 10, N = 50, P = 100, fixed_effect = None, fit_intercept = False):
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

def likelihood_compare(comp, theta, num_sim = 10, N = 50, P = 100,
                       fixed_effect = None, fit_intercept = False, likefcn=['YYT_ZZT','YYT_ZTZ','YTY_ZZT','YTY_ZTZ']):
    Q = comp.shape[0]
    s = sqrt(exp(theta[comp]))
    z = np.zeros((num_sim*2,))

    # Prepare output structure
    n_param = theta.shape[0]
    df = {}
    for i in range(n_param):
        df.update({'theta' + str(i): z})
    df.update({'R2_Pcm':z,'time_Pcm':z,'likefcn':z})
    T = pd.DataFrame(df)

    # Prepare Ridge regression
    M = pcm.regression.RidgeDiag(comp, fit_intercept = fit_intercept)

    for i in range(num_sim):
        # Make training data
        U = np.random.normal(0,1,(Q,P))
        U = U * s.reshape((Q,1))
        Z = np.random.normal(0,1,(N,Q))
        Y = Z @ U + np.random.normal(0,sqrt(exp(theta[-1])),(N,P))

        for j,lf in enumerate(likefcn):
            idx = i*len(likefcn) + j
            # PCM regression
            t0 = time.perf_counter()
            M.optimize_regularization(Z,Y,like_fcn = lf)
            T.loc[idx,'time_Pcm'] = time.perf_counter() - t0
            for k in range(n_param):
                T.loc[idx,'theta'+str(k)] = M.theta_[k]
            T.loc[idx,'likefcn'] = lf
            T.loc[idx,'iter'] = M.optim_info['iter']
    return T
