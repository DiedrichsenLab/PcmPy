"""
Bootstrap routines for Group-level model fitting
Included are fast version of the routine using cross-block correlation estimation

Author: joern.diedrichsen@googlemail.com
"""

import numpy as np
import PcmPy as pcm

def bootstrap_group_corr(D,M,fixed_effect=None,n_bootstr=1000,fit_scale=False,boot_indx=None,verbose=False):
    """ Bootstrap the group correlation estimate of the PCM model

    Parameters:
        D (list of pcm.Datasets):
            List of datasets for each subject
        M (pcm.Model):
            PCM correlation model to fit
        fixed_effect (array-like or list of array-like):
            Optional fixed effect to include in the model. if only one is given, it is assumed to be the same for all subjects, otherwise a subject-specific list.
        n_bootstr (int):
            Number of bootstrap samples to draw (default: 1000)
        fit_scale (bool):
            Whether to fit the scale parameter of the model (default: False)
        boot_indx (ndarray):
            Optional pre-defined bootstrap indices (shape: n_subj x n_bootstr)
        verbose (bool):
            Whether to print progress information during bootstrapping (default: False)
    Returns:
        r_boot (ndarray):
            Bootstrap distribution of the group correlation estimate (shape: n_bootstr,)
        fSNR_boot (ndarray):
            Bootstrap distribution of the group fSNR estimate (shape: n_bootstr,)
        boot_indx (ndarray):
            The bootstrap indices used for the resampling (shape: n_subj x n_bootstr
    """
    n_subj=len(D)

    # Get fit on the original data
    T,theta0 = pcm.fit_model_group(D,M,fixed_effect=fixed_effect,fit_scale=False,add_prior=False,verbose=0)

    # bootstrap group estimate
    if boot_indx is None:
        boot_indx = np.random.choice(n_subj,size=(n_subj,n_bootstr),replace=True)
    r_boot = np.zeros((n_bootstr,))
    fSNR_boot = np.zeros((n_bootstr,))
    if verbose:
        print(f"Bootstrapping {n_bootstr} samples: ")
    for i in range(n_bootstr):
        if verbose and (i % 50 == 0):
            print(f"Bootstrap sample {i}/{n_bootstr}")
        d = [D[j] for j in boot_indx[:,i]]
        if isinstance(fixed_effect,list):
            X = [fixed_effect[j] for j in boot_indx[:,i]]
        else:
            X = fixed_effect
        tt,th=pcm.fit_model_group(d,M,fixed_effect=X,fit_scale=False,add_prior=False,verbose=0)
        th_g,_ = pcm.group_to_individ_param(th[0],M,n_subj)
        r_boot[i]= M.get_correlation(th_g).mean()
        fSNR_boot[i] = M.get_fSNR(th_g, separate=False).mean()
    return r_boot,fSNR_boot,boot_indx
