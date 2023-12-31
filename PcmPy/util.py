#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Collection of different utility functions

@author: jdiedrichsen
"""

import numpy as np
from numpy import sum,mean,trace,sqrt, zeros, ones
from PcmPy.matrix import indicator, pairwise_contrast
from scipy.linalg import solve, pinv
from scipy.spatial import procrustes
from scipy.spatial.distance import squareform
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

def G_to_dist(G):
    """Transforms a second moment matrix
    into a squared Euclidean matrix (mostly for visualization)

    Args:
        G (ndarray): 2d or 3d array of second moment matrices
    """
    K = G.shape[1]
    C=pairwise_contrast(np.arange(K))
    if G.ndim == 2:
        d = np.diag(C @ G @ C.T)
        D = squareform(d)
    else:
        raise(NameError('3d not implemented yet'))
    return D

def classical_mds(G,contrast=None,align=None,thres=0):
    """Calculates a low-dimensional projection of a G-matrix
    That preserves the relationship of different conditions
    Equivalent to classical MDS.
    If contrast is given, the method becomes equivalent to dPCA,
    as it finds the representation that maximizes the variance acording to this contrast.
    Developement: If `align` is given, it performs Procrustes alignment of the result to a given V within the found dimension

    Args:
        G (ndarray): KxK second moment matrix
        contrast (ndarray): Contrast matrix to optimize for. Defaults to None.
        align (ndarry): A different loading matrix to which to align
        thres (float): Cut off eigenvalues under a certain value
    Returns:
        W (ndarray): Loading of the K different conditions on main axis
        Glam (ndarray): Variance explained by each axis
    """
    G = (G + G.T)/2
    Glam, V = eigh(G)
    Glam = np.flip(Glam,axis=0)
    V = np.flip(V,axis=1)

    # Kill eigenvalues smaller than threshold
    Glam[Glam<thres]=0
    W = V * np.sqrt(Glam)

    # When aligning - use the center and scale of the target
    # As the standard for both
    if align is not None:
        align_m = align.mean(axis=0)
        align_std = align - align_m
        align_s = trace(align_std.T@align_std)
        align_std = align_std / sqrt(align_s)
        A,W1,disp = procrustes(align_std,W)
        W = W1 * sqrt(align_s) + align_m
        Glam = np.diag(W.T @ W)
    return W,Glam

def check_grad(fcn,theta0,delta=0.0001):
    """Checks the gradient of a function around a value for theta

    Args:
        fcn (function): needs to return criterion and derivative
        theta0 (ndarray): Vector of parameters
        delta (float): Step size for gradient. Defaults to 0.0001.
    """
    x,dx = fcn(theta0)
    for i in range(len(theta0)):
        theta = theta0.copy()
        theta[i]=theta0[i]+delta
        xp,_ = fcn(theta)
        theta[i]=theta0[i]-delta
        xn,_ = fcn(theta)
        est_grad = (xp-xn)/2/delta
        print('Estimate gradient:')
        print(est_grad )
        print('Returned gradient:')
        print(dx[i])
        print('Error:',((est_grad-dx[i])**2).sum())

