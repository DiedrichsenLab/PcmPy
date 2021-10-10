#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Collection of different utility functions

@author: jdiedrichsen
"""

import numpy as np
from PcmPy.matrix import indicator
from scipy.linalg import solve, pinv
from numpy.linalg import eigh

def est_G_crossval(Y, Z, part_vec, X=None, S=None):
    """
    Obtains a crossvalidated estimate of G
    Y = Z @ U + X @ B + E, where var(U) = G

    Parameters:
        Y (numpy.ndarray)
            Activity data
        Z (numpy.ndarray)
            2-d: Design matrix for conditions / features U
            1-d: condition vector
        part_vec (numpy.ndarray)
            Vector indicating the partition number
        X (numpy.ndarray)
            Fixed effects to be removed
        S (numpy.ndarray)

    Returns:
        G_hat (numpy.ndarray)
            n_cond x n_cond matrix
        Sig (numpy.ndarray)
            n_cond x n_cond noise estimate per block

    """

    N , n_channel = Y.shape
    part = np.unique(part_vec)
    n_part = part.shape[0]

    # Make Z into a design matrix
    if Z.ndim == 1:
        Z = indicator(Z)
    n_cond = Z.shape[1]

    # Allocate memory
    A = np.zeros((n_part,n_cond,n_channel)) # Allocate memory
    Bp = np.zeros((n_cond,n_channel))
    G = np.zeros((n_part,n_cond,n_cond)) # Allocate memory

    # If fixed effects are given, remove
    if X is not None:
        Y -= X @ pinv(X) @ Y

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

def make_pd(G,thresh = 1e-10):
    """
    Enforces that G is semi-positive definite by setting small eigenvalues to minimal value

    Parameters:
        G   (square 2d-np.array)
            estimated KxK second momement matrix
        thresh (float)v
            threshold for increasing small eigenvalues
    Returns:
        Gpd (square 2d-np.array)
            semi-positive definite version of G

    """

    G = (G + G.T) / 2 # Symmetrize
    Glam, V = eigh(G)
    Glam[Glam < thresh] = thresh # rectify small eigenvalues
    G_pd = V @ np.diag(Glam) @ V.T
    return G_pd
