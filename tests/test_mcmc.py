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

def sim_correlation_sample(n_items=1,
                     n_part=6,
                     log_signal=-2,
                     corr=1.0,
                     n_subj=16,
                     rem_mpat=False,
                     var_param='separate'):
    """ Simulates mcmc sampling from a group model 
    Here we use a correlation model
    """
    rng = np.random.default_rng()
    # Make Flexible model:
    Mflex = pcm.CorrelationModel('flex',num_items=n_items,
                    corr=None,cond_effect=False)
    # Make models under the two hypotheses:
    Mtrue = pcm.CorrelationModel('r0',num_items=n_items,
                    corr=corr,cond_effect=False)
    if var_param=='common':
        Mflex.common_param=[True,True,True]
        Mtrue.common_param=[True,True]
    else:
        Mflex.common_param=[False,False,True]
        Mtrue.common_param=[False,False]

    cond_vec,part_vec = pcm.sim.make_design(n_items*2,n_part)
    D = pcm.sim.make_dataset(Mtrue, [0,0],
        cond_vec,
        part_vec=part_vec,
        n_sim=n_subj,
        signal=np.exp(log_signal),
        rng=rng)
    # calculate the relevant stats:
    T,theta,dFdhh = pcm.fit_model_group(D,Mflex,fixed_effect=None,fit_scale=False,return_second_deriv=True)
    
    th,l = pcm.sample_model_group(D,Mflex,fixed_effect=None,fit_scale=False)

    return T,theta,Mflex,th,l


if __name__ == '__main__':
    # np.seterr(all='raise')
    # np.seterr(over='ignore')
    # a=np.array([1e50,1e100,3,4])
    # b=np.exp(a)
    sim_correlation_sample()
    pass

