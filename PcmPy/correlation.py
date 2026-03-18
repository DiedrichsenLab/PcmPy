"""
Functions for Cross-block correlation estimation as laid out in
Diedrichsen et al. 2026
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.spatial.distance import squareform

def cond_to_item(cond_vec):
    """Takes a cond_vec and assigns the first half
    to condition1, and the second half to condition 2
    cond can be any integer number - sorted by size and unique
    Example:
    cond_vec  = [1,2,3,4,5,6]
    con_vec =   [0,0,0,1,1,1]
    item_vec =  [0,1,2,0,1,2]
    Args:
        cond_vec (ndarray): original condition vector
    Returns:
        con_vec (ndarray): Condition [0,1] vector
        item_vec (ndarray): Item vector
    """
    cond,cond_vec_new = np.unique(cond_vec,return_inverse=True)
    n_cond = len(cond)
    m_cond = int(n_cond/2)
    con_vec  = (cond_vec_new>=m_cond).astype(int)
    item_vec = (cond_vec_new-con_vec*m_cond).astype(int)
    return con_vec,item_vec

def multi_to_uni_item(Y,cond_vec,part_vec,
                        rem_mval=False,
                        rem_mpat=False):
    """ Takes correlation patterns with multi-item
     then for correlation analysis by stretching them out
    Args:
        Y (ndarray): _description_
        cond_vec (ndarray): original condition vector (either 1 or 0 based)
        part_vec (ndarray): original partition vector
        rem_mval (bool): Remove the mean value (across voxel). False.
        rem_mpat (bool): Remove the mean pattern (across condition).
    Returns:
        X (ndarray): 2(conditions) x n_part x n_item*n_vox tensor
    """
    part = np.unique(part_vec)
    cond = np.unique(cond_vec)
    n_part = len(part)
    n_cond = len(cond)

    con_vec,item_vec = cond_to_item(cond_vec)
    # Remove mean val
    if rem_mval:
        Y = Y - Y.mean(axis=1).reshape(-1,1)

    # Loop over participants and assemble data
    n_item=max(item_vec)+1
    X = np.zeros((2,n_part,n_item*Y.shape[1]))
    for i,p in enumerate(part):
        for c in range(2):
            y = Y[(part_vec==p) & (con_vec==c),:]
            if rem_mpat:
                y = y - y.mean(axis=0)
            X[c,i,:]=y.flatten()
    return X

def combine_epsilon(var,var_cr,n_part):
    """Combines the epsilon estimates for noise and recomputes signal co-variance

    Args:
        var (_type_): _description_
        var_cr (_type_): _description_
        n_part (_type_): _description_
    """
    N = np.expand_dims(n_part,1)
    sig_e= (var-var_cr)*N # Estimate of noise covariance
    sig_e=sig_e.mean(axis=1,keepdims=True)
    sig_s = var - sig_e/N # Estimate of signal covariance
    return sig_s,sig_e

def calc_sufficient_stats(Y,rem_mpat=False,rem_mval=False):
    """returns Sufficient statistics for correlation analysis
    Args:
        Y (list): List of PCM datasets
        rem_mpat (bool): Remove mean pattern in condition (False)
        rem_mval (bool): Remove mean value (corr or cosang). Defaults to False.
    Returns:
        var: Variance of X and Y mean pattern
        var_cr: Variance estimate for X and Y estimated from cross-block
        cov: Covariance between X and Y mean pattern
        cov_cr: Covariance between X and Y estimated only from cross-block
        n_part: Number of partitions
    """
    n_subj = len(Y)
    var = np.zeros((n_subj,2))
    cov = np.zeros((n_subj,))
    var_cr = np.zeros((n_subj,2))
    cov_cr = np.zeros((n_subj,))
    n_part = np.zeros((n_subj,),dtype=int)

    for i,D in enumerate(Y):
        cond_vec = D.obs_descriptors['cond_vec']
        part_vec = D.obs_descriptors['part_vec']
        X = multi_to_uni_item(D.measurements,
                cond_vec,part_vec,
                rem_mpat=rem_mpat,rem_mval=rem_mval)
        _,n_part[i],P = X.shape

        COV = X[0] @ X[1].T / P
        I = np.eye(n_part[i])==0
        cov[i]=COV.mean()
        cov_cr[i]=COV[I].mean()

        for j in range(2):
            VAR= X[j] @ X[j].T / P
            var[i,j]=VAR.mean()
            var_cr[i,j]=VAR[I].mean()
    return var,var_cr,cov,cov_cr,n_part

def get_corr_raw(var,cov):
    """ Gives the uncorrected correlation estimate from sufficent stats
    Args:
        var: Variance of X and Y mean pattern
        cov: Covariance between X and Y mean pattern
    Returns:
        r_unc: cov/sqrt(var(x) * var(y))
    """
    denom = np.sqrt(var[:,0]*var[:,1])
    r_unc = cov / denom
    return r_unc

def calc_corr_raw(Y,rem_mpat=False,rem_mval=False):
    """ Calculates the uncorrected correlation estimate
    Args:
        Y (list): List of PCM datasets
        rem_mpat (bool): Remove mean pattern in condition (False)
        rem_mval (bool): Remove mean value (corr or cosang). Defaults to False.
    Returns:
        r_unc: Uncorrected correlation estimate
    """
    var,var_cr,cov,cov_cr = calc_sufficient_stats(Y,rem_mpat,rem_mval)
    r_unc = get_corr_raw(var,cov)
    return r_unc

def get_corr_adj(var,var_cr,cov,n_part,single_eps=True,negvar=0):
    """ Calculates the adjusted correlation estimate from sufficent stats
    uses the combined epsilon estimate if single_eps is True.
    Args:
        var: Variance of X and Y mean pattern
        var_cr: Variance estimate for X and Y estimated from cross-block
        cov: Covariance between X and Y mean pattern
        negvar: What to do when variance estimates are negative: np.nan: exclude, 0: take sign
    Returns:
        r_adj: cov/sqrt(var(x) * var(y))
        sig_s: Signal variance estimate
        sig_e: Noise variance estimate
    """
    if single_eps:
        sig_s,sig_e = combine_epsilon(var,var_cr,n_part)
    else:
        sig_s = var_cr
        sig_e = (var-var_cr)*n_part.reshape(-1,1) # Estimate of noise covariance
    sig_s[sig_s<=0]=negvar
    r_adj = cov/np.sqrt(sig_s[:,0]*sig_s[:,1])
    r_adj = np.clip(r_adj,-1,1)
    return r_adj,sig_s,sig_e

def get_corr_adj_group(var,var_cr,cov,n_part,negvar=0):
    """ Calculates the adjusted group correlation estimate individual sufficent stats
    Args:
        var: Variance of X and Y mean pattern
        var_cr: Variance estimate for X and Y estimated from cross-block
        cov: Covariance between X and Y mean pattern
    Returns:
        r_adj: cov/sqrt(var(x) * var(y))
        sig_s: Signal variance estimate
        sig_e: Noise variance estimate
    """
    sig_s,sig_e = combine_epsilon(var,var_cr,n_part)
    msig_s = sig_s.mean(axis=0,keepdims=True)
    mcov = cov.mean(axis=0,keepdims=True)
    msig_s[msig_s<0]=negvar
    r_adj = mcov/np.sqrt(msig_s[:,0]*msig_s[:,1])
    r_adj = np.clip(r_adj,-1,1)
    return r_adj,msig_s,sig_e.mean(axis=0,keepdims=True)

def calc_corr_adj(Y,rem_mpat=False,rem_mval=False,single_eps=True,negvar=0):
    """ Calculates corrected correlation coefficient

    Args:
        Y (list): List of PCM datasets
        rem_mpat (bool): Remove mean pattern in condition (False)
        rem_mval (bool): Remove mean value (corr or cosang). Defaults to False.
        single_eps (bool): Use single epsilon estimate all conditions. Defaults to True.
        negvar (float): Value to replace negative variance estimates. Defaults to 0.
        reestimate_at_bound (bool): Reestimate the variance at the correlation bound. Defaults to True.
    Returns:
        r_adj: Adjusted correlation coefficient (r/sqrt(rel1*rel2))
        sig_s: Signal variance estimate
        sig_e: Noise variance estimate
    """
    var,var_cr,cov,cov_cr,n_part = calc_sufficient_stats(Y,rem_mpat,rem_mval)
    r_adj,sig_s,sig_e = get_corr_adj(var,var_cr,cov,n_part,single_eps,negvar)
    return r_adj,sig_s,sig_e


