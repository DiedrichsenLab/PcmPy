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

def group_by_condition(Y, Z, part_vec, axis=-1):
    """
    Averages activity patterns from individual trials by condition within each partition
    Parameters:
        Y (numpy.ndarray)
            Activity data
        Z (numpy.ndarray)
            2-d: Design matrix for conditions / features U
            1-d: condition vector
        part_vec (numpy.ndarray)
            Vector indicating the partition number
        axis (int)
            Trial axis
    Returns:

    """

    part = np.unique(part_vec)
    n_part = part.shape[0]

    # Make Z into a design matrix
    if Z.ndim == 1:
        Z = indicator(Z)
    n_cond = Z.shape[1]

    # Move trial axis to front for easier indexing
    Y = np.moveaxis(Y, axis, 0)  # shape (T, ...)

    Y_grp, Z_grp = [], []

    for p in part:
        idx = np.where(part_vec == p)[0]
        Yp = Y[idx]  # shape (n_trials_in_part, ...)
        Zp = Z[idx, :]  # shape (n_trials_in_part, n_cond)

        # Check which conditions are present in this partition
        present = Zp.sum(axis=0) > 0

        # Preallocate output (with NaNs for missing conditions)
        Yc = np.full((n_cond,) + Yp.shape[1:], np.nan)

        # Compute only for present conditions
        denom = Zp[:, present].sum(axis=0, keepdims=True)
        denom[denom == 0] = 1
        weights = Zp[:, present] / denom

        w = weights.T[(...,) + (None,) * (Yp.ndim - 1)]
        Yc[present] = np.sum(w * Yp[None, ...], axis=1)

        Y_grp.append(Yc)
        Z_grp.append(np.eye(n_cond))

    Y_grp = np.vstack(Y_grp)  # shape (n_part, n_cond, D1, D2, ...)
    Z_grp = np.vstack(Z_grp)

    part_vec_grp = np.repeat(part, n_cond)

    return Y_grp, Z_grp, part_vec_grp


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

    N, n_channel = Y.shape
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
    Yr = Y.copy()
    if X is not None:
        Yr -= X @ pinv(X) @ Yr

    # Estimate condition means within each run and crossvalidate
    for i in range(n_part):
        #  Left-out partition
        indxA = part_vec == part[i]
        Za = Z[indxA,:]
        # Za = Za[:,any(Za,1)] # restrict to regressors that are not all 0
        Ya = Yr[indxA,:]

        # remainder of conditions
        indxB = part_vec != part[i]
        Zb = Z[indxB, :]
        # Zb    = Zb(:,any(Zb,1));    % Restrict to regressors that are not
        Yb = Yr[indxB, :]

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


def est_G(Y, Z, part_vec, X=None):
    N, n_channel = Y.shape
    part = np.unique(part_vec)
    n_part = part.shape[0]

    # Make Z into a design matrix
    if Z.ndim == 1:
        Z = indicator(Z)
    n_cond = Z.shape[1]

    # Allocate memory
    A = np.zeros((n_part, n_cond, n_channel))
    G = np.zeros((n_part, n_cond, n_cond))  # Allocate memory

    # If fixed effects are given, remove
    Yr = Y.copy()
    if X is not None:
        Yr -= X @ pinv(X) @ Yr

    # Estimate condition means within each run and crossvalidate
    for i in range(n_part):
        indxA = part_vec == part[i]
        Za = Z[indxA, :]
        Ya = Yr[indxA, :]
        a = pinv(Za) @ Ya
        A[i, :, :] = a
        G[i, :, :] = a @ a.T / n_channel  # normalised to the number of voxels
    G = np.mean(G, axis=0)

    # bias term from across-partition variability (same as your Sig code)
    R = A - A.mean(axis=0)
    Sig = np.zeros((n_cond, n_cond))
    for i in range(n_part):
        Sig += (R[i] @ R[i].T) / n_channel
    Sig /= (n_part - 1)  # (n_cond x n_cond)
    return G, Sig



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
        M = C[None, :, :] @ (G @ C.T)  # (T, P, P)
        d = np.diagonal(M, axis1=-2, axis2=-1)  # (T, P)
        D = np.stack([squareform(di) for di in d], axis=0)  # (T, K, K)
    return D


def G_to_cosine(G):
    """
    Computes cosine similarity between all pairs of conditions from a positive semi-definite second moment matrix G.

    Parameters:
        G (ndarray): K x K second moment matrix (assumed PSD).

    Returns:
        cosine_sim (ndarray): K x K cosine similarity matrix.
    """

    if np.isnan(G).any() or np.isinf(G).any():
        raise ValueError("G contains NaN or inf values")

    # check if positive semi-definite
    eigvals = np.linalg.eigvalsh(G)
    if ~np.all(eigvals >= -1e-10):
        G = make_pd(G)

    G = 0.5 * (G + G.T)  # Ensure symmetry
    diag = np.diag(G)
    norm = np.sqrt(np.outer(diag, diag))
    cosine = G / norm
    return cosine


def classical_mds(G,contrast=None,align=None,thres=0):
    """Calculates a low-dimensional projection of a G-matrix
    That preserves the relationship of different conditions
    Equivalent to classical MDS.
    If contrast is given, the method becomes equivalent to dPCA,
    as it finds the representation that maximizes the variance acording to this contrast.
    Developement: If `align` is given, it performs Procrustes alignment of the result to a given V within the found dimension

    Args:
        G (ndarray): KxK second moment matrix
        contrast (ndarray): QxK Contrast matrix to optimize for. Defaults to None.
        align (ndarry): A different loading matrix to which to align
        thres (float): Cut off eigenvalues under a certain value
    Returns:
        W (ndarray): Loading of the K different conditions on main axis
        Glam (ndarray): Variance explained by each axis
    """

    # If contrast is given, find the projection that maximizes the variance
    if contrast is not None:
        # Project G into contrast subspace
        H = contrast @ pinv(contrast)
        Gc = H @ G @ H.T  # Projected G

        # Compute eigendecomposition in contrast subspace
        eigvals, V = eigh(Gc)
        eigvals = np.flip(eigvals, axis=0)
        V = np.flip(V, axis=1)

        # Project eigenvectors back to original space
        W = H.T @ V * np.sqrt(eigvals)
        Glam = eigvals

    else:
        G = (G + G.T) / 2
        Glam, V = eigh(G)
        Glam = np.flip(Glam, axis=0)
        V = np.flip(V, axis=1)

        # Kill eigenvalues smaller than threshold
        Glam[Glam < thres] = 0
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

