#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions for data simulation from PCM-models
@author: jdiedrichsen
"""

import numpy as np
import scipy.stats as ss
import scipy.linalg as sl
from scipy.spatial.distance import squareform
from PcmPy import matrix
from PcmPy import dataset

def make_design(n_cond, n_part):
    """
    Makes simple fMRI design with n_cond, each measures n_part times

    Args:
        n_cond (int):          Number of conditions
        n_part (int):          Number of partitions

    Returns:
        Tuple (cond_vec, part_vec)
        cond_vec (np.ndarray): n_obs vector with condition
        part_vec (np.ndarray): n_obs vector with partition

    """
    p = np.array(range(0, n_part))
    c = np.array(range(0, n_cond))
    cond_vec = np.kron(np.ones((n_part,)), c)   # Condition Vector
    part_vec = np.kron(p, np.ones((n_cond,)))    # Partition vector
    return(cond_vec, part_vec)


def make_dataset(model, theta, cond_vec, n_channel=30, n_sim=1,
                signal=1, noise=1, signal_cov_channel=None, noise_cov_channel=None,noise_cov_trial=None, use_exact_signal=False, use_same_signal=False,
                part_vec = None, rng = None):
    """
    Simulates a fMRI-style data set

    Args:
        model (PcmPy.Model):
            the model from which to generate data
        theta (numpy.ndarray):
            vector of parameters (one dimensional)
        cond_vec (numpy.ndarray):
            RSA-style model: vector of experimental conditions
            Encoding-style: design matrix (n_obs x n_cond)
        n_channel (int):
            Number of channels (default = 30)
        n_sim (int):
            Number of simulation with the same signal (default = 1)
        signal (float):
            Signal variance (multiplied by predicted G)
        signal_cov_channel(numpy.ndarray):
            Covariance matrix of signal across channels
        noise (float):
            Noise variance
        noise_cov_channel(numpy.ndarray):
            Covariance matrix of noise (default = identity)
        noise_cov_trial(numpy.ndarray):
            Covariance matrix of noise across trials
        use_exact_signal (bool):
            Makes the signal so that G is exactly as specified (default: False)
        use_same_signal (bool):
            Uses the same signal for all simulation (default: False)
        part_vec (np.array):
            Optional partition that is added to the data set obs_descriptors
        rng (np.random.default_rng):
            Optional random number generator object to pass specific state
    Returns:
        data (list):              List of pyrsa.Dataset with obs_descriptors
    """

    # Get the model prediction and build second moment matrix
    # Note that this step assumes that RDM uses squared Euclidean distances
    G, _ = model.predict(theta)

    # If no random number generator is passed - make one
    if rng is None:
        rng = np.random.default_rng()

    # Make design matrix
    if (cond_vec.ndim == 1):
        Zcond = matrix.indicator(cond_vec)
    elif (cond_vec.ndim == 2):
        Zcond = cond_vec
    else:
        raise(NameError("cond_vec needs to be either vector or design matrix"))
    n_obs, n_cond = Zcond.shape

    # If signal_cov_channel is given, precalculate the cholesky decomp
    if (signal_cov_channel is not None):
        if (signal_cov_channel.shape is not (n_channel, n_channel)):
            raise(NameError("Signal covariance for channels needs to be \
                             n_channel x n_channel array"))
        signal_chol_channel = np.linalg.cholesky(signal_cov_channel)
    else:
        signal_chol_channel = None

    # If noise_cov_channel is given, precalculate the cholinsky decomp
    if (noise_cov_channel is not None):
        if (noise_cov_channel.shape is not (n_channel, n_channel)):
            raise(NameError("noise covariance for channels needs to be \
                             n_channel x n_channel array"))
        noise_chol_channel = np.linalg.cholesky(noise_cov_channel)
    else:
        noise_chol_channel = None

    # If noise_cov_trial is given, precalculate the cholinsky decomp
    if (noise_cov_trial is not None):
        if (noise_cov_trial.shape is not (n_channel, n_channel)):
            raise(NameError("noise covariance for trials needs to be \
                             n_obs x n_obs array"))
        noise_chol_trial = np.linalg.cholesky(noise_cov_trial)
    else:
        noise_chol_trial = None

    # Generate the signal - here same for all simulations
    if (use_same_signal):
        true_U = make_signal(G, n_channel, use_exact_signal, signal_chol_channel)

    # Generate noise as a matrix normal, independent across partitions
    # If noise covariance structure is given, it is assumed that it's the same
    # across different partitions
    obs_des = {"cond_vec": cond_vec,"part_vec": part_vec}
    des = {"signal": signal, "noise": noise,
           "model": model.name, "theta": theta}
    dataset_list = []
    for i in range(0, n_sim):
        # If necessary - make a new signal
        if (use_same_signal == False):
            true_U = make_signal(G, n_channel, use_exact_signal, signal_chol_channel,rng=rng)
        # Make noise with normal distribution - allows later plugin of other dists
        epsilon = rng.uniform(0, 1, size=(n_obs, n_channel))
        epsilon = ss.norm.ppf(epsilon) * np.sqrt(noise)
        # Now add spatial and temporal covariance structure as required
        if (noise_chol_channel is not None):
            epsilon = epsilon @ noise_chol_channel
        if (noise_chol_trial is not None):
            epsilon = noise_chol_trial @ epsilon
        # Assemble the data set
        data = Zcond @ true_U * np.sqrt(signal) + epsilon
        datas = dataset.Dataset(data, obs_descriptors=obs_des, descriptors=des)
        dataset_list.append(datas)
    return dataset_list

def make_signal(G, n_channel,make_exact=False, chol_channel=None,rng = None):
    """
    Generates signal exactly with a specified second-moment matrix (G)

    Args:
        G(np.array)        : desired second moment matrix (ncond x ncond)
        n_channel (int)    : Number of channels
        make_exact (bool)  : Make the signal so the second moment matrix is exact
                             (default: False)
        chol_channel: Cholensky decomposition of the signal covariance matrix
                             (default: None - makes signal i.i.d.)
        rng (np.random.default_rng):
                            Optional random number generator object to pass specific state

    Returns:
        np.array (n_cond x n_channel): random signal

    """
    # Generate the true patterns with exactly correct second moment matrix
    n_cond = G.shape[0]

    # If no random number generator is passed - make one
    if rng is None:
        rng = np.random.default_rng()

    # We use two-step procedure allow for different distributions later on
    true_U = rng.uniform(0, 1, size=(n_cond, n_channel))
    true_U = ss.norm.ppf(true_U)
    # Make orthonormal row vectors
    if (make_exact):
        E = true_U @ true_U.transpose()
        L = np.linalg.cholesky(E)
        true_U = np.linalg.solve(L, true_U) * np.sqrt(n_channel)
    # Impose spatial covariance matrix
    if (chol_channel is not None):
        true_U = true_U @ chol_channel
    # Now produce data with the known second-moment matrix
    # Use positive eigenvectors only
    # (cholesky does not work with rank-deficient matrices)
    lam, V = np.linalg.eigh(G)
    lam[lam < 1e-15] = 0
    lam = np.sqrt(lam)
    chol_G = V * lam.reshape((1, V.shape[1]))
    true_U = (chol_G @ true_U)
    return true_U