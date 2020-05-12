#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Collection of different utility functions

@author: jdiedrichsen
"""

import numpy as np
import PcmPy as pcm
from scipy.linalg import solve, pinv

def est_G_crossval(Y, Z, part_vec, X=None, S=None):
    """
    Obtains a crossvalidated estimate of G
        Y = Z @ U + X @ B + E
        Where var(U) = G

    Parameters:
        Y (numpy.ndarray)
            Activity data
        Z (numpy.ndarray)
            Design matrix for eximation of U)
            should the function ignore zero negative entries
            in the index_vector? Default: false
        part_vec (numpy.ndarray)
            Vector indicating the partition number
        X (numpy.ndarray)
        S (numpy.ndarray)
    Returns:
        G_hat (numpy.ndarray)
            n_cond x n_cond matrix
        Sig_hat (numpy.ndarray)

    """

    N , n_channel = Y.shape
    part = np.unique(part_vec)
    n_part = part.shape[0]
    n_cond = Z.shape[1]

    A = np.zeros((n_part,n_cond,n_channel)) # Allocate memory 
    Bp = np.zeros((n_cond,n_channel))
    G = np.zeros((n_part,n_cond,n_cond)) # Allocate memory 

    # Estimate condition means within each run and crossvalidate 
    for i in range(n_part):
        #  Left-out partition 
        indxA = part_vec == part[i]
        Za = Z[indxA,:] 
        # Za = Za[:,any(Za,1)] # restrict to regressors that are not all 0
        Ya = Y[indxA,:]

        # remainder of conditions 
        indxB = part_vec != part[i]
        Zb = Z[indxB, :]
        # Zb    = Zb(:,any(Zb,1));    % Restrict to regressors that are not  
        Yb = Y[indxB, :]

        a = pinv(Za) @ Ya
        b = pinv(Zb) @ Yb
        A[i,:,:] = a
        G[i,:,:] = a @ b.T / n_channel # normalised to the number of voxels 
    G = np.mean(G,axis = 0)

    # Estimate noise covariance matrix
    Sig = np.zeros((n_part,n_cond,n_cond))
    R = A-np.sum(A,axis=0) / n_part
    for i in range(n_part): 
        Sig[i,:,:] = R[i,:,:] @ R[i,:,:].T / n_channel
    Sig = np.sum(Sig, axis=0) / (n_part-1)
    return [G, Sig]