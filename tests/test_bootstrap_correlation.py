#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_indicator

@author: jdiedrichsen
"""

import PcmPy as pcm
import numpy as np


def group_bootstrap():
    Mtrue = pcm.CorrelationModel('corr', num_items=1, corr=0.7, cond_effect=False)
    # Create the design. In this case it's 2 conditions, across 8 runs (partitions)
    n_cond=2
    n_part = 8
    cond_vec, part_vec = pcm.sim.make_design(n_cond, n_part)

    # Starting from the true model above, generate 30 datasets/participants
    # with relatively low signal-to-noise ratio (0.06:1)
    # Note that signal variance is drawn from a gamma distribution with shape=2 and scale=0.03 (mean 0.06)
    # The noise is by default set to 1.
    # For replicability, we are using a fixed random seed (100)
    n_subj = 30
    rng = np.random.default_rng(seed=102)
    signal = np.random.gamma(shape=2,scale=0.03,size=(n_subj,))
    D = pcm.sim.make_dataset(model=Mtrue, \
        theta=[0,0], \
        cond_vec=cond_vec, \
        part_vec=part_vec, \
        n_sim=n_subj, \
        signal=signal,\
        rng=rng)

    # Now make the flexible model
    Mflex = pcm.CorrelationModel("flex", num_items=1, corr=None, cond_effect=False)

    T_gr, theta_gr = pcm.fit_model_group(D, Mflex, fixed_effect=None, fit_scale=True, verbose=False)
    T_gr.head()
    # Add the group estimate as a horizontal line
    theta_g,_= pcm.group_to_individ_param(theta_gr[0],Mflex,20)
    r_group = Mflex.get_correlation(theta_g)

    r_boot,fSNR_boot,_= pcm.bootstrap_group_corr(D,Mflex,fixed_effect=None,n_bootstr=200,verbose=True)
    pass

if __name__ == '__main__':
    group_bootstrap()