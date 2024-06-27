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
import cProfile

def sim_correlation_sample(n_items=1,
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
    Mflex = pcm.CorrelationModel('flex',num_items=n_items,
                    corr=None,cond_effect=False)
    # Make models under the two hypotheses:
    Mtrue = pcm.CorrelationModel('r0',num_items=n_items,
                    corr=corr,cond_effect=False)
    if var_param=='common':
        Mflex.common_param=[True,True,True]
        corr_indx = 2
        Mtrue.common_param=[True,True]
    else:
        Mflex.common_param=[False,False,True]
        corr_indx = 0
        Mtrue.common_param=[False,False]

    # Make the correlation models from -1 to 1
    nsteps = 21
    M=[]
    corr_list = np.linspace(0,1,nsteps)
    for r in corr_list:
        m=pcm.CorrelationModel(f"{r:0.2f}",num_items=1,corr=r,cond_effect=False)
        if var_param=='common':
            m.common_param=[True,True]
        else:
            m.common_param=[False,False] # For group fit, allow different variances
        M.append(m)

    cond_vec,part_vec = pcm.sim.make_design(n_items*2,n_part)
    D = pcm.sim.make_dataset(Mtrue, [0,0],
        cond_vec,
        part_vec=part_vec,
        n_sim=n_subj,
        signal=np.exp(log_signal),
        n_channel=50,
        rng=rng)

    # Approximate the posterior using the profile log-likelihood
    Tg,_ = pcm.inference.fit_model_group(D,M,fixed_effect=None,fit_scale=False)
    like_list = Tg.likelihood.values.sum(axis=0)
    like_list = like_list-like_list.mean()
    prop_list = exp(like_list)
    prop_list = prop_list/prop_list.sum()
    # Fit the group model and get second derivative:
    T,theta,dFdhh = pcm.fit_model_group(D,Mflex,fixed_effect=None,fit_scale=False,return_second_deriv=True)
    sd = np.sqrt(1/(dFdhh[0]+0.01))*0.2
    # cProfile.run('pcm.sample_model_group(D,Mflex,fixed_effect=None,fit_scale=False,theta0=theta[0],proposal_sd=sd)')
    # Sample the posterior
    sample_param = {'n_samples':8000,'burn_in':100}
    th,l = pcm.sample_model_group(D,Mflex,fixed_effect=None,fit_scale=False,theta0=theta[0],proposal_sd=sd,sample_param=sample_param)

    r = (exp(2*th[corr_indx])-1)/(exp(2*th[corr_indx])+1)

    plt.figure()
    plt.subplot(3,1,1)
    plt.plot(th[corr_indx])
    plt.subplot(3,1,2)
    plt.plot(l)
    plt.subplot(3,1,3)
    sb.histplot(r,stat='density',bins=corr_list)
    plt.plot(corr_list,prop_list/(corr_list[1]-corr_list[0]),'r')
    return T,theta,Mflex,th,l

def ll_gaussian(theta,prec,return_deriv=0): 
    lloglik = -0.5 * theta.T@ prec @ theta  # Nultivariate Gaussian prior
    if return_deriv==0:
        return (-lloglik,)
    dloglik = - prec @ theta 
    if return_deriv==1:
        return (-lloglik, -dloglik)
    ddloglik = -prec
    return (-lloglik, -dloglik, -ddloglik)

def test_mcmc_gaussian(n_dim=4):
    """ Test the MCMC sampling with a Gaussian prior
    """
    prec = np.eye(n_dim)*0.1
    proposal_sd = 1./np.sqrt(np.diag(prec))
    theta0 = np.zeros((n_dim,))
    fcn = lambda x: ll_gaussian(x,prec,return_deriv=0)
    th,l=pcm.mcmc(theta0,fcn,proposal_sd=proposal_sd,n_samples=10000)
    pass 






if __name__ == '__main__':
    test_mcmc_gaussian(n_dim=20)
    # np.seterr(all='raise')
    # np.seterr(over='ignore')
    # a=np.array([1e50,1e100,3,4])
    # b=np.exp(a)

    pass

