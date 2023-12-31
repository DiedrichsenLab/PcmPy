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


def test_gradient_correlation(corr=0.7,signal=1.0,rand_seed=1):
    rng=np.random.default_rng(rand_seed)
    cond_vec,part_vec = pcm.sim.make_design(2,5)

    Mflex = pcm.CorrelationModelRefprior("flex",num_items=1,corr=None,cond_effect=False)
    Mtrue = pcm.CorrelationModel('corr',num_items=1,corr=corr,cond_effect=False)
    D = pcm.sim.make_dataset(Mtrue, [0,0], cond_vec,part_vec=part_vec,
                                 n_sim=1,
                                 signal=signal,
                                 rng=rng)
    Z,X,YY,n_channel,Noise,G_hat = pcm.inference.set_up_fit(D[0],
                                                fixed_effect = None,
                                                noise_cov = None)
    Mflex.set_theta0(G_hat)
    th0 = np.append(Mflex.theta0,0)
    th0[2] = 1.5
    #  Now do the fitting, using the preferred optimization routine
    fcn = lambda x: pcm.inference.likelihood_individ(x, Mflex, YY, Z, X=X,
                                Noise = Noise, fit_scale = False, return_deriv = 1,n_channel=n_channel)
    pcm.util.check_grad(fcn,th0,delta=0.001)
    pass


if __name__ == '__main__':
    # np.seterr(all='raise')
    test_gradient_correlation()

    # np.seterr(over='ignore')
    # a=np.array([1e50,1e100,3,4])
    # b=np.exp(a)
    pass

