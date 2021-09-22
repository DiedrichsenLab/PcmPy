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

def get_corr(X,cond_vec):
    p1 = X[cond_vec==0,:].mean(axis=0)
    p2 = X[cond_vec==1,:].mean(axis=0)
    return np.corrcoef(p1,p2)[0,1]

def do_sim(corr,signal=np.linspace(0,8,20),n_sim=50): 
    M = pcm.CorrelationModel('corr',num_items=1,corr=corr,cond_effect=False)
    G,dG = M.predict([0,0])
    cond_vec,part_vec = pcm.sim.make_design(2,2)
    Lcorr = []
    Lsign = []
    for s in signal:
        D = pcm.sim.make_dataset(M, [0,-0.5], cond_vec, n_sim=n_sim, signal=s)
        for i in range(n_sim):
            Lcorr.append(get_corr(D[i].measurements,cond_vec))
            Lsign.append(s)
    S = pd.DataFrame({'r_naive':Lcorr,'signal':Lsign})
    return S

if __name__ == '__main__':
    D = do_sim(0.7)
    pass
