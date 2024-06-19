#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_convergence with challenging data

@author: jdiedrichsen
"""

import pandas as pd
import PcmPy as pcm
import numpy as np

def sim_estimation_individ(corr,
                    log_signal=np.linspace(-4,1,10),
                    n_items=1,
                    n_part=5,
                    n_vox=30,
                    n_sim=50,
                    rem_mpat=False,
                    randseed=None):
    """ Produce a simulation with a a set number of subjects for each log-signal level

    Args:
        corr (float): true correlation value
        log_signal (array): Defaults to np.linspace(0,2,10).
        n_items (int, optional): Number of items. Defaults to 1.
        n_part (int, optional): Number of partitions (N). Defaults to 5.
        n_vox (int, optional): Number of voxels (P). Defaults to 30.
        n_sim (int, optional): Number of datasets per signal level. Defaults to 50.
        rem_mpat (bool, optional): Remove mean pattern. Defaults to False.
        randseed (int, optional): Random seed. Defaults to None.

    Returns:
        results (pd): Dataframe with the results
    """
    M = pcm.CorrelationModel('corr',num_items=n_items,corr=corr,cond_effect=False)
    cond_vec,part_vec = pcm.sim.make_design(n_items*2,n_part)
    rng = np.random.default_rng(randseed)
    S = pd.DataFrame()

    # Make flexible PCM model for getting the maximum likelhood estimate
    Mflex = pcm.CorrelationModel('flex',num_items=n_items,
                    corr=None,cond_effect=False)

    # Loop over different levels of log-signal:
    for s in log_signal:
        d = {}
        theta = [s,s] # [log(sigma2_x),log(sigma2_y)]
        D = pcm.sim.make_dataset(M, theta,
                        cond_vec,
                        part_vec=part_vec,
                        n_sim=n_sim,
                        signal=1,
                        rng=rng,
                        n_channel=n_vox)
        d['r_true']=[corr]*n_sim
        d['signal']=[np.exp(s)]*n_sim
        d['log_signal']=[s]*n_sim
        d['n_vox']=[n_vox]*n_sim
        d['n_items']=[n_items]*n_sim

        # Get maximum likelihood estimate
        T,theta_hat=pcm.fit_model_individ(D,Mflex,fixed_effect=None,fit_scale=False)
        d['r_pcm']=Mflex.get_correlation(theta_hat[0])
        d['var_x_pcm']=np.exp(theta_hat[0][0,:])
        d['var_y_pcm']=np.exp(theta_hat[0][1,:])
        d['var_eps_pcm']=T['noise']['flex'].copy()
        # Another way to get the Sigma_eps
        # d['var_eps_pcm1']=np.exp(theta_hat[0][3,:])

        S = pd.concat([S,pd.DataFrame(d)],ignore_index=True)
    return S

def do_sim():
    log_signal=np.linspace(-8,-6,1)
    T=sim_estimation_individ(0.8,n_items=1,n_sim=500,log_signal=log_signal)
    pass

if __name__=="__main__":
    do_sim()
    pass
