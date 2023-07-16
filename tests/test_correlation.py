#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_indicator

@author: jdiedrichsen
"""
# Import necessary libraries
import PcmPy as pcm
from PcmPy import sim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from numpy import exp, sqrt
import matplotlib.pyplot as plt


def do_curve_sim(corr=0.7,signal=0.1,n_sim=20,rand_seed=1):
    rng=np.random.default_rng(rand_seed)
    # Make the design in this case it's 2 runs, 2 conditions!
    cond_vec,part_vec = pcm.sim.make_design(2,2)
    # Generate different models from 0 to 1
    M=[]
    for r in np.linspace(-1,1,21):
        M.append(pcm.CorrelationModel(f"{r:0.1f}",num_items=1,corr=r,cond_effect=False))
    Mflex = pcm.CorrelationModel("flex",num_items=1,corr=None,cond_effect=False)
    M.append(Mflex)
    # For each simulation scenario, get different
    Mtrue = pcm.CorrelationModel('corr',num_items=1,corr=corr,cond_effect=False)
    D = pcm.sim.make_dataset(Mtrue, [0,0], cond_vec,part_vec=part_vec,
                                 n_sim=n_sim,
                                 signal=signal[i],
                                 rng=rng)
    T,theta = pcm.inference.fit_model_individ(D,M,fixed_effect=None,fit_scale=False)
    r = M[-1].get_correlation(theta[-1])
    L = T.likelihood.iloc[:,:-1]
    LL = L.values-L.values.mean(axis=1).reshape(-1,1)
    corr = np.array(T.likelihood.columns[:-1]).astype(float)
    plt.plot(corr,LL.T)
    return T,theta,M

def do_prior_sim(corr=0.7,signal=0.1,
                 n_sim=20,
                 rand_seed=1,
                 prior=0,
                 prec=np.linspace(0,1,10)):
    rng=np.random.default_rng(rand_seed)
    # Make the design in this case it's 2 runs, 2 conditions!
    cond_vec,part_vec = pcm.sim.make_design(2,2)
    # Generate different models from 0 to 1
    M=[]
    for p in prec:
        m = pcm.CorrelationModel(f"{p:0.1f}",num_items=1,corr=None,cond_effect=False)
        m.prior_mean=np.array([0,0,prior])
        m.prior_prec=np.array([0,0,p])
        M.append(m)
    # For each simulation scenario, get different
    Mtrue = pcm.CorrelationModel('corr',num_items=1,corr=corr,cond_effect=False)
    D = pcm.sim.make_dataset(Mtrue, [0,0], cond_vec,part_vec=part_vec,
                                 n_sim=n_sim,
                                 signal=signal,
                                 rng=rng)
    T,theta = pcm.inference.fit_model_individ(D,M,fixed_effect=None,fit_scale=False)
    th=np.vstack([t[0,:] for t in theta]) # Variance parameter
    r=np.vstack([M[0].get_correlation(t) for t in theta])
    plt.subplot(2,1,1)
    plt.plot(prec,r)
    plt.subplot(2,1,2)
    plt.plot(prec,th)
    return T,r,M



if __name__ == '__main__':
    # np.seterr(all='raise')
    T,theta,M = do_prior_sim()

    # np.seterr(over='ignore')
    # a=np.array([1e50,1e100,3,4])
    # b=np.exp(a)
    pass

