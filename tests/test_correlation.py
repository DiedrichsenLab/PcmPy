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




def do_sim(corr=[0.7],signal=[0.1],n_sim=20):
    # Make the design in this case it's 2 runs, 2 conditions!
    cond_vec,part_vec = pcm.sim.make_design(2,2)
    # Generate different models from 0 to 1
    M=[]
    for r in np.linspace(0,1,11):
        M.append(pcm.CorrelationModel(f"{r:0.1f}",num_items=1,corr=r,cond_effect=False))
    Mflex = pcm.CorrelationModel("flex",num_items=1,corr=None,cond_effect=False)
    M.append(Mflex)
    # For each simulation scenario, get different
    for i,r in enumerate(corr):
        Mtrue = pcm.CorrelationModel('corr',num_items=1,corr=r,cond_effect=False)
        D = pcm.sim.make_dataset(Mtrue, [0,0], cond_vec,part_vec=part_vec,n_sim=n_sim, signal=signal[i])
        T,theta = pcm.inference.fit_model_individ(D,M,fixed_effect=None,fit_scale=False)
    return T,theta,M


if __name__ == '__main__':
    # np.seterr(all='raise')
    T,theta,M = do_sim()
    r = M[-1].get_correlation(theta[-1])
    L = T.likelihood.iloc[:,:-1]
    # np.seterr(over='ignore')
    # a=np.array([1e50,1e100,3,4])
    # b=np.exp(a)
    pass

