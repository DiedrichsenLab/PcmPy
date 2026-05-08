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
import seaborn as sb
from numpy import exp, sqrt
import copy

def sim_correlation(n_items=1,
                     n_part=6,
                     log_signal=-1,
                     corr=0.7,
                     n_subj=16,
                     rem_mpat=False,
                     var_param='separate'):
    """ Simulates mcmc sampling from a group model
    Here we use a correlation model
    """
    rng = np.random.default_rng()
    # Make Flexible model:
    # Make models under the two hypotheses:
    Mtrue = pcm.CorrelationModel('r0',num_items=n_items,
                    corr=corr,cond_effect=False)

    cond_vec,part_vec = pcm.sim.make_design(n_items*2,n_part)
    D = pcm.sim.make_dataset(Mtrue, [0,0],
        cond_vec,
        part_vec=part_vec,
        n_sim=n_subj,
        signal=np.exp(log_signal),
        n_channel=50,
        rng=rng)

    return D, cond_vec,part_vec, Mtrue

def fit_model(n_subj=20):
    D,cond_vec,part_vec,Mtrue = sim_correlation(n_subj=n_subj)
    D1 = copy.deepcopy(D)
    for i in range(len(D)):
        D1[i].measurements = D1[i].measurements - np.mean(D1[i].measurements,axis=0)
    Mflex =  pcm.CorrelationModel('r1',num_items=1,corr=None,cond_effect=False)
    M0 =  pcm.CorrelationModel('r1',num_items=1,corr=0,cond_effect=False)
    T1,theta1 = pcm.fit_model_individ(D,Mflex,fixed_effect=None,fit_scale=False)
    T2,theta2 = pcm.fit_model_individ(D1,Mflex,fixed_effect=None,fit_scale=False)
    T3,theta3 = pcm.fit_model_individ(D,Mflex,fixed_effect=np.ones((12,1)),fit_scale=False)
    T4,theta4 = pcm.fit_model_individ(D,Mflex,fixed_effect='block',fit_scale=False)
    T1['method']=['None']*n_subj
    T2['method']=['Remove']*n_subj
    T3['method']=['ReML']*n_subj
    T4['method']=['Block']*n_subj
    T1['var1']=np.exp(theta1[0][0,:])
    T1['var2']=np.exp(theta1[0][1,:])
    T1['sig']=np.exp(theta1[0][3,:])
    T1['corr']=Mflex.get_correlation(theta1[0])
    T2['var1']=np.exp(theta2[0][0,:])
    T2['var2']=np.exp(theta2[0][1,:])
    T2['sig']=np.exp(theta2[0][3,:])
    T2['corr']=Mflex.get_correlation(theta2[0])
    T3['var1']=np.exp(theta3[0][0,:])
    T3['var2']=np.exp(theta3[0][1,:])
    T3['corr']=Mflex.get_correlation(theta3[0])
    T3['sig']=np.exp(theta3[0][3,:])
    T4['var1']=np.exp(theta4[0][0,:])
    T4['var2']=np.exp(theta4[0][1,:])
    T4['corr']=Mflex.get_correlation(theta4[0])
    T4['sig']=np.exp(theta4[0][3,:])
    T = pd.concat([T1,T2,T3,T4])
    return T


if __name__ == '__main__':
    T=fit_model()
    plt.subplot(2,2,1)
    sb.barplot(T,x='method',y='var1')
    plt.subplot(2,2,2)
    sb.barplot(T,x='method',y='var2')
    plt.subplot(2,2,3)
    sb.barplot(T,x='method',y='sig')
    plt.subplot(2,2,4)
    sb.barplot(T,x='method',y='corr')
    pass



