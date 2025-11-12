import numpy as np
from numpy import sum, diag, log, eye, exp, trace, einsum
import PcmPy as pcm


def CKA_individ(theta, M, YY, Z, X=None,
                return_deriv=1): # return deriv 1 for 
    '''
    Compute CKA (cosine similarity) between two matrices

    Returns:
        CKA (double):
            CKA (cosine similarity) between the two matrices

        dCKAdtheta (1d-np.array):
            First derivative of CKA in respect to the fitted parameters

        ddCKAdtheta2 (2d-np.array):
            Second derivative of CKA in respect to the fitted parameters
    '''

    N = YY.shape[0]
    # Q = Z.shape[1]
    n_param = theta.shape[0]

    # Get Model parameters, G-matrix and derivative of G-matrix in respect to parameters
    model_params = theta[range(M.n_param)]
    G,dGdtheta = M.predict(model_params)

    # Center the matrices:
    H = np.eye(N) - np.ones((N, N)) / N
    YY_centered = H @ YY @ H
    G_centered = H @ G @ H

    # column-vectorize the centered matrices:
    a = YY_centered.flatten()
    b = G_centered.flatten()

    # perform CKA:
    cka = (a.T @ b) / (np.linalg.norm(a) * np.linalg.norm(b))

    if return_deriv == 0:
        return cka
    
    # ==============================================================
    # ===== compute the first derivative of CKA in respect to theta:
    # ==============================================================
    # dCKAdtheta = dCKAdG . dGdtheta -> vectorize: dCKAdtheta = dCKAdb . dbdtheta
    dCKAdb = a/(np.linalg.norm(a) * np.linalg.norm(b)) - \
            cka * b/(np.linalg.norm(b)**2)

    dCKAdtheta = np.zeros((n_param,))
    for i in range(n_param):
        dGdtheta_i = dGdtheta[i].flatten()
        dCKAdtheta[i] = np.dot(dCKAdb, dGdtheta_i)

    if return_deriv == 1:
        return cka, dCKAdtheta

    # ==============================================================
    # ==== compute the second derivative of CKA in respect to theta:
    # ==============================================================
    if return_deriv == 2:
        raise NotImplementedError("Second derivative of CKA not implemented yet.")


def fit_model_individ(Data, M, fixed_effect='block', fit_scale=False,
                    scale_prior = 1e3, noise_cov=None, algorithm=None,
                    optim_param={}, theta0=None, verbose = True,
                    return_second_deriv=False,add_prior=False):
    '''
    DESCRIPTION...

    Paramters:

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


