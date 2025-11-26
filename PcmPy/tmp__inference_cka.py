import numpy as np
from numpy import sum, diag, log, eye, exp, trace, einsum
import PcmPy as pcm
import pandas as pd
import scipy.optimize as opt

def _CKA_individ(theta, M, G_est): # return deriv 1 for 
    '''
    Compute CKA (cosine similarity) between two matrices

    Returns:
        CKA (double):
            CKA (cosine similarity) between the two matrices

        dCKAdtheta (1d-np.array):
            First derivative of CKA in respect to the fitted parameters

    '''

    N = G_est.shape[0]
    # Q = Z.shape[1]
    n_param = theta.shape[0]

    # Get Model parameters, G-matrix and derivative of G-matrix in respect to parameters
    model_params = theta[range(M.n_param)]
    G,dGdtheta = M.predict(model_params)

    # Center the matrices:
    H = np.eye(N) - np.ones((N, N)) / N
    G_est_centered = H @ G_est @ H
    G_centered = H @ G @ H

    # column-vectorize the centered matrices:
    a = G_est_centered.flatten()
    b = G_centered.flatten()

    # perform CKA:
    epsilon = 1e-10
    cka = (a.T @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + epsilon)
    
    # ==============================================================
    # ===== compute the first derivative of CKA in respect to theta:
    # ==============================================================
    # dCKAdtheta = dCKAdG . dGdtheta -> vectorize: dCKAdtheta = dCKAdb . dbdtheta
    dCKAdb = a/(np.linalg.norm(a) * np.linalg.norm(b) + epsilon) - \
            cka * b/(np.linalg.norm(b)**2 + epsilon)

    dCKAdtheta = np.zeros((n_param,))
    for i in range(n_param):
        dGdtheta_i = dGdtheta[i].flatten()
        dCKAdtheta[i] = np.dot(dCKAdb, dGdtheta_i)

    return -cka, -dCKAdtheta

    # ==============================================================
    # ==== compute the second derivative of CKA in respect to theta:
    # ==============================================================
    # if return_deriv == 2:
    #     raise NotImplementedError("Second derivative of CKA not implemented yet.")


def fit_CKA_individ(Data, M, fixed_effect='block', theta0=None, verbose = True):
    '''
    DESCRIPTION...
    
    Parameters:
    
    Returns:


    '''
    # Get the number of subjects
    if type(Data) is list:
        n_subj = len(Data)
    else:
        n_subj = 1
        Data = [Data]

    # Make sure fixed effects are subject specific
    if type(fixed_effect) is not list:
        fixed_effect = [fixed_effect] * n_subj

    # Get the number of models
    if type(M) in [list,pcm.model.ModelFamily]:
        n_model = len(M)
    else:
        n_model = 1
        M = [M]
        if (theta0 is not None) and (not isinstance(theta0,list)):
            theta0 = [theta0]

    # Get model names and determine fitted params
    m_names = []
    for m in M:
        m_names.append(m.name)
        # Which parameter should be fitted
        if hasattr(m,'fit_param'):
            m.fit_param = np.array(m.fit_param)
        else:
            m.fit_param = np.ones((m.n_param,), dtype=np.bool_)

    # Preallocate output structures
    iterab = [['CKA','iterations'],m_names]
    index = pd.MultiIndex.from_product(iterab, names=['variable', 'model'])
    T = pd.DataFrame(np.zeros((n_subj, n_model * 2)), columns=index)
    theta = [None] * n_model

    # Loop over subject and models and provide inidivdual fits
    for s in range(n_subj):
        # Get the cross-validated data G matrix:
        G_est,_ =  pcm.est_G_crossval(Data[s].measurements,
                                    Data[s].obs_descriptors['cond_vec'],
                                    Data[s].obs_descriptors['part_vec'],
                                    X=pcm.matrix.indicator(Data[s].obs_descriptors['part_vec']))
        
        # loop over models:
        for i,m in enumerate(M):
            if verbose:
                print('Fitting Subj',s,'model',i)
            # Get starting guess for theta0 is not provided
            if (theta0 is None) or (len(theta0) <= i) or (theta0[i].shape[1]<s):
                th0  = m.get_theta0(G_est)
                # th0 = np.ones(th0.shape)  # use ones as starting guess
            else:
                th0 = theta0[i]

            #  Now minimize the -CKA i.e., maximize CKA:
            res = opt.minimize(
                            fun=_CKA_individ,
                            x0=th0,
                            args=(m, G_est),
                            jac=True,          # tells SciPy that function returns (CKA, dCKAdtheta)
                            method="L-BFGS-B",
                            options={'maxiter': 1000,'gtol': 1e-3,'ftol': 1e-3,})
            theta_hat = res.x
            
            # Record results
            T.loc[s,('CKA',m_names[i])] = -res.fun
            T.loc[s,('iterations',m_names[i])] = res.nit

            # Record theta parameters
            if theta[i] is None:
                theta[i] = np.zeros((theta_hat.shape[0],n_subj))
            theta[i][:,s] = theta_hat

    return T, theta

def fit_CKA_group_crossval(Data, M, theta0, fixed_effect='block', verbose=True):
    """
    Description...

    Parameters:

    Returns:

    """
    # Get the number of subjects
    if type(Data) is list:
        n_subj = len(Data)
    else:
        n_subj = 1
        Data = [Data]

    # Get the number of models
    if type(M) in [list,pcm.model.ModelFamily]:
        n_model = len(M)
    else:
        n_model = 1
        M = [M]

    # Get model names and common parameters
    m_names = []
    for m in M:
        m_names.append(m.name)
        if hasattr(m,'common_param'):
            m.common_param = np.array(m.common_param)
        else:
            m.common_param = np.ones((m.n_param,), dtype=np.bool_)

    # Preallocate output structures
    # Preallocate output structures
    iterab = [['CKA','CKA_fit','iterations'],m_names]
    index = pd.MultiIndex.from_product(iterab, names=['variable', 'model'])
    T = pd.DataFrame(np.zeros((n_subj, n_model * 3)), columns=index)
    theta = [None] * n_model

    # get G_est for each subject
    n_conditions = len(np.unique(Data[0].obs_descriptors['cond_vec']))
    G_est = np.zeros((n_subj, n_conditions, n_conditions))
    for s in range(n_subj):
        # Get the cross-validated data G matrix:
        G_est[s,:,:],_ =  pcm.est_G_crossval(Data[s].measurements,
                                        Data[s].obs_descriptors['cond_vec'],
                                        Data[s].obs_descriptors['part_vec'],
                                        X=pcm.matrix.indicator(Data[s].obs_descriptors['part_vec']))

    # Loop over subject and models and provide inidivdual fits
    for s in range(n_subj):
        notS = np.arange(n_subj) != s # Get indices of training group
        
        # Get the cross-validated data G matrix:
        G_loo = np.mean(G_est[notS,:,:], axis=0)
        G_ind = G_est[s,:,:]

        # loop over models:
        for i,m in enumerate(M):
            if verbose:
                print('Fitting LOO Subj',s,'model',i)
            # Get starting guess for theta0 is not provided
            if (theta0 is None) or (len(theta0) <= i) or (theta0[i].shape[1]<s):
                th0  = m.get_theta0(G_loo)
            else:
                th0 = theta0[i]

            #  Now minimize the -CKA i.e., maximize CKA on loo group:
            res = opt.minimize(
                            fun=_CKA_individ,
                            x0=th0,
                            args=(m, G_loo),
                            jac=True,          # tells SciPy that function returns (CKA, dCKAdtheta)
                            method="L-BFGS-B",
                            options={'maxiter': 1000,'gtol': 1e-3,'ftol': 1e-3})
            theta_hat = res.x
            
            # get CKA on left-out subject:
            cka_ind,_ = _CKA_individ(theta_hat, m, G_ind) # remember the function returns -CKA
            T['CKA_fit',m_names[i]][s] = -res.fun
            T['CKA',m_names[i]][s] = -cka_ind 
            T['iterations',m_names[i]][s] = res.nit

            # Record theta parameters
            if theta[i] is None:
                theta[i] = np.zeros((theta_hat.shape[0],n_subj))
            theta[i][:,s] = theta_hat
    
    # # estimate noise ceiling:
    # ceil_high = np.zeros((n_subj,))
    # ceil_low = np.zeros((n_subj,))
    # G_est_avg = np.mean(G_est, axis=0)
    # for s in range(n_subj):
    #     # high ceiling:
    #     ceil_high_tmp,_ = _CKA_individ(theta_hat, m, G_est_avg)
    #     ceil_high[s] = -ceil_high_tmp

    #     # low ceiling:
    #     notS = np.arange(n_subj) != s # Get indices of training group
    #     G_loo = np.mean(G_est[notS,:,:], axis=0)
    #     ceil_low_tmp,_ = _CKA_individ(theta_hat, m, G_loo)
    #     ceil_low[s] = -ceil_low_tmp

    return T, theta