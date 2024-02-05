"""
Inference module for PCM toolbox with main functionality for model fitting and evaluation.
@author: jdiedrichsen
"""

import numpy as np
from numpy.linalg import solve, eigh, cholesky, cond
from numpy import sum, diag, log, eye, exp, trace, einsum
import pandas as pd
import PcmPy as pcm
from PcmPy.model import IndependentNoise, BlockPlusIndepNoise
from PcmPy.optimize import newton,mcmc


def likelihood_individ(theta, M, YY, Z, X=None,
                       Noise = IndependentNoise(),
                       n_channel=1,
                       fit_scale=False,
                       scale_prior = 1e3,
                       return_deriv=0):
    """Negative Log-Likelihood of the data and derivative in respect to the parameters

    Parameters:
        theta (np.array):
            Vector of (log-)model parameters - these include model, signal, and noise parameters
        M (PcmPy.model.Model):
            Model object with predict function
        YY (2d-np.array):
            NxN Matrix of outer product of the activity data (Y*Y')
        Z (2d-np.array):
            NxQ Design matrix - relating the trials (N) to the random effects (Q)
        X (np.array):
            Fixed effects design matrix - will be accounted for by ReML
        Noise (pcm.Noisemodel):
            Pcm-noise mode to model block-effects (default: IndepenentNoise)
        n_channel (int):
            Number of channels
        fit_scale (bool):
            Fit a scaling parameter for the model (default is False)
        scale_prior (float):
            Prior variance for log-normal prior on scale parameter
        return_deriv (int):
            0: Only return negative loglikelihood (default)
            1: Return first derivative
            2: Return first and second derivative

    Returns:
        negloglike (double):
            Negative log-likelihood of the data under a model

        dLdtheta (1d-np.array):
            First derivative of negloglike in respect to the fitted parameters

        ddLdtheta2 (2d-np.array):
            Second derivative of negloglike in respect to the fitted parameters

    """

    N = YY.shape[0]
    Q = Z.shape[1]
    n_param = theta.shape[0]

    # Get Model parameters, G-matrix and derivative of G-matrix in respect to parameters
    model_params = theta[range(M.n_param)]
    prior, dprior, ddprior = M.get_prior(model_params)
    G,dGdtheta = M.predict(model_params)

    # Get the scale parameter and scale G by it
    if fit_scale:
        scale_param = theta[M.n_param]
        indx_scale = M.n_param # Index of scale parameter
    else:
        scale_param = 0
    Gs = G * exp(scale_param)

    # Get the noise model parameters
    noise_params = theta[M.n_param+fit_scale:]

    # Apply the matrix inversion lemma. The following statement is the same as
    # V   = (Z*Gs*Z' + S(noiseParam));
    # iV  = pinv(V);
    Gs = (Gs + Gs.T) / 2 # Symmetrize
    Glambda, GU = eigh(Gs)
    idx = Glambda > (10e-10) # Find small eigenvalues
    Zu = Z @ GU[:, idx]
    iS = Noise.inverse(noise_params)
    if type(iS) is np.float64:
        matrixInv = (diag(1 / Glambda[idx]) / iS + Zu.T @ Zu)
        iV = (eye(N) - Zu @ solve(matrixInv, Zu.T)) * iS
    else:
        matrixInv = (diag(1 / Glambda[idx])  + Zu.T @ iS @ Zu)
        iV = iS - iS @ Zu @ solve(matrixInv,Zu.T) @ iS
    # For ReML, compute the modified inverse iVr
    if X is not None:
        iVX   = iV @ X
        iVr   = iV - iVX @ solve(X.T @ iVX, iVX.T)
    else:
        iVr = iV

    # Computation of (restricted) likelihood
    ldet = -2 * sum(log(diag(cholesky(iV)))) # Safe computation
    llik = -n_channel / 2 * ldet - 0.5 * einsum('ij,ij->',iVr, YY)
    if X is not None:
        # P/2 log(det(X'V^-1*X))
        llik -= n_channel * sum(log(diag(cholesky(X.T @ iV @X)))) #

    # add the log-normal prior to the parameters
    if fit_scale:
        llik -= scale_param**2 / (2 * scale_prior) # Add prior
    llik += prior # Add prior

    # If no derivative - exit here
    if return_deriv == 0:
        return (-llik,) # Return as tuple for consistency

    # Calculate the first derivative
    A = iVr @  Z
    B = YY @ iVr
    iVdV = []

    # Get the quantity iVdV = inv(V)dVdtheta for model parameters
    for i in range(M.n_param):
        iVdV.append(A @ dGdtheta[i,:,:] @ Z.T * exp(scale_param))

    # Get iVdV for scaling parameter
    if fit_scale:
        iVdV.append(A @ G @ Z.T * exp(scale_param))

    # Get iVdV for Noise parameters
    for j in range(Noise.n_param):
        dVdtheta = Noise.derivative(noise_params,j)
        if type(dVdtheta) is np.float64:
            iVdV.append(iVr * dVdtheta)
        else:
            iVdV.append(iVr @ dVdtheta)

    # Based on iVdV we can get he first derivative
    dLdtheta = np.zeros((n_param,))
    for i in range(n_param):
        dLdtheta[i] = -n_channel / 2 * trace(iVdV[i]) + 0.5 * einsum('ij,ij->',iVdV[i], B) # Trace(A@B.T)

    # Add log-normal prior to the model and possible scale parameters
    dLdtheta[range(M.n_param)] += dprior
    if fit_scale:
        dLdtheta[indx_scale] -= scale_param / scale_prior

    # If only first derivative, exit here
    if return_deriv == 1:
        return (-llik, -dLdtheta)

    # Calculate expected second derivative
    d2L = np.zeros((n_param,n_param))
    for i in range(n_param):
        for j in range(i, n_param):
            d2L[i, j] = -n_channel / 2 * einsum('ij,ji->',iVdV[i],iVdV[j]) # Trace(A@B)
            d2L[j, i] = d2L[i, j]

    # Add log-normal prior to the model and possible scale parameters
    d2L[0:M.n_param, 0:M.n_param] += ddprior
    if fit_scale:
        d2L[indx_scale, indx_scale] -= 1 / scale_prior

    if return_deriv == 2:
        return (-llik, -dLdtheta, -d2L)
    else:
        raise NameError('return_deriv needs to be 0, 1 or 2')

def likelihood_group(theta, M, YY, Z, X=None,
                       Noise=IndependentNoise(),
                       n_channel=1, fit_scale=True, scale_prior=1e3,
                       return_deriv=0,return_individ=False):
    """Negative Log-Likelihood of group data and derivative in respect to the parameters

    Parameters:
        theta (np.array):
            Vector of (log-)model parameters consisting of common model parameters (M.n_param or sum of M.common_param) +
            participant-specific parameters (iterated by subject):
            individ model param (not in common_param),
            scale parameter
            noise parameters
        M (pcm.Model):
            Model object
        YY (List of np.arrays):
            List of NxN Matrix of outer product of the activity data (Y*Y')
        Z (List of 2d-np.array):
            NxQ Design matrix - relating the trials (N) to the random effects (Q)
        X (List of np.array):
            Fixed effects design matrix - will be accounted for by ReML
        Noise (List of pcm.Noisemodel):
            Pcm-noise model (default: IndependentNoise)
        n_channel (List of int):
            Number of channels
        fit_scale (bool):
            Fit a scaling parameter for the model (default is False)
        scale_prior (float):
            Prior variance for log-normal prior on scale parameter
        return_deriv (int):
            0: Only return negative likelihood
            1: Return first derivative
            2: Return first and second derivative (default)
        return_individ (bool):
            return individual likelihoods instead of group likelihood
        return_deriv (int):
            0:None, 1:First, 2: second

    Returns:
        negloglike:
            Negative log-likelihood of the data under a model

        dLdtheta (1d-np.array)
            First derivative of negloglike in respect to the parameters

        ddLdtheta2 (2d-np.array)
            Second derivative of negloglike in respect to the parameters

    """
    n_subj = len(YY)
    n_param = theta.shape[0]
    # preallocate the arrays
    nl = np.zeros((n_subj,))
    dFdh = np.zeros((n_subj,n_param))
    dFdhh = np.zeros((n_subj,n_param,n_param))

    # Get the mapping to individual subjects
    ths,indx=group_to_individ_param(theta,M,n_subj)

    # Loop over subjects and get individual likelihoods
    for s in range(n_subj):
        # Get individual likelihood
        res = likelihood_individ(ths[:,s], M, YY[s], Z[s], X[s],
                       Noise[s], n_channel[s], fit_scale = fit_scale, scale_prior = scale_prior, return_deriv = return_deriv)
        nl[s] = res[0]
        if return_deriv>0:
            dFdh[s, indx[:,s]] = res[1]
        if return_deriv==2:
            ixgrid = np.ix_([s],indx[:,s],indx[:,s])
            dFdhh[ixgrid] = res[2]

    # Integrate over subjects
    if return_individ:
        ra = [nl]
    else:
        ra = [np.sum(nl, axis=0)]
    if return_deriv > 0:
        ra.append(np.sum(dFdh,axis=0)) # First derivative
    if return_deriv > 1:
        ra.append(np.sum(dFdhh,axis=0)) # Second derivative
    return ra


def group_to_individ_param(theta,M,n_subj):
    """ Takes a vector of group parameters and rearranges them
    To make it conform to theta you would get back from a individual fit

    Args:
        theta (nd.array): Vector of group parameters
        M (pcm.Model): PCM model
        n_subj (int): Number of subjects
    Returns:
        theta_indiv (ndarray): n_params x n_subj Matrix of group parameters
    """
    if hasattr(M,'common_param'):
        common = np.array(M.common_param)
    else:
        common = np.ones((M.n_param,), dtype=np.bool_)

    n_gparam = len(theta)
    n_common = common.sum()
    n_indiv = np.floor_divide(n_gparam-n_common,n_subj)
    if np.remainder(n_gparam-n_common,n_subj)>0:
        raise(NameError(f'Group parameters vector is not the right size.'))
    pindx = (np.arange(n_subj*n_indiv)+n_common).reshape(n_subj,n_indiv).T
    indx =  np.zeros((n_common + n_indiv,n_subj),int)
    for p,i in enumerate(np.where(common)[0]):
        indx[i,:]=p
    for p,i in enumerate(np.where(np.logical_not(common))[0]):
        indx[i,:]=pindx[p,:]
    # Signal and noise parameters
    indx[M.n_param:,:]=pindx[(M.n_param-n_common):,:]
    theta_indiv= theta[indx]
    return theta_indiv,indx

def fit_model_individ(Data, M, fixed_effect='block', fit_scale=False,
                    scale_prior = 1e3, noise_cov=None, algorithm=None,
                    optim_param={}, theta0=None, verbose = True,
                    return_second_deriv=False):
    """Fits Models to a data set individually.

    The model parameters are all individually fit.

    Parameters:
        Data (pcm.Dataset or list of pcm.Datasets):
            List data set has partition and condition descriptors
        M (pcm.Model or list of pcm.Models):
            Models to be fitted on the data sets
        fixed effect:
            None, 'block', or nd-array. Default ('block') adds an intercept for each partition
        fit_scale (bool):
            Fit a additional scale parameter for each subject? Default is set to False.
        scale_prior (float):
            Prior variance for log-normal prior on scale parameter
        algorithm (string):
            Either 'newton' or 'minimize' - provides over-write for model specific algorithms
        noise_cov:
            None (i.i.d), 'block', or optional specific covariance structure of the noise
        optim_param (dict):
            Additional paramters to be passed to the optimizer
        theta0 (list of np.arrays):
            List of starting values (same format as return argument theta)
        verbose (bool):
            Provide printout of progress? Default: True

    Returns:
        T (pandas.dataframe):
            Dataframe with the fields:
            SN:                 Subject number
            likelihood:         log-likelihood
            scale:              Scale parameter (if fitscale = 1)-exp(theta_s)
            noise:              Noise parameter- exp(theta_eps)
            run:                Run parameter (if run = 'random')
            iterations:         Number of interations for model fit
            time:               Elapsed time in sec

        theta (list of np.arrays):
            List of estimated model parameters, each a
            #params x #numSubj np.array

        G_pred (list of np.arrays):
            List of estimated G-matrices under the model

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
    iterab = [['likelihood','noise','iterations'],m_names]
    index = pd.MultiIndex.from_product(iterab, names=['variable', 'model'])
    T = pd.DataFrame(np.zeros((n_subj, n_model * 3)), columns=index)
    theta = [None] * n_model
    if return_second_deriv:
        dLdhh = [None] * n_model
    # Determine optimal algorithm for each of the models
    # M = pcm.optimize.best_algorithm(M,algorithm)

    # Loop over subject and models and provide inidivdual fits
    for s in range(n_subj):
        Z,X,YY,n_channel,Noise,G_hat = set_up_fit(Data[s],
                                                fixed_effect = fixed_effect,
                                                noise_cov = noise_cov)
        for i,m in enumerate(M):
            if verbose:
                print('Fitting Subj',s,'model',i)
            # Get starting guess for theta0 is not provideddf
            if (theta0 is None) or (len(theta0) <= i) or (theta0[i].shape[1]<s):
                th0  = m.get_theta0(G_hat)
                if (fit_scale):
                    G_pred, _ = m.predict(th0)
                    scale0 = get_scale0(G_pred, G_hat)
                    th0 = np.concatenate((th0,scale0))
                th0 = np.concatenate((th0,Noise.get_theta0(Data[s].measurements, Z, X)))
            else:
                th0 = theta0[m][:,s]
            # Get the parameters to be fitted
            fit_param = np.r_[m.fit_param,np.ones(th0.shape[0]-m.n_param,dtype=bool)]
            #  Now do the fitting, using the preferred optimization routine
            if (m.algorithm=='newton'):
                fcn = lambda x: likelihood_individ(x, m, YY, Z, X=X,
                                Noise = Noise, fit_scale = fit_scale, scale_prior = scale_prior, return_deriv = 2,n_channel=n_channel)
                th, l, INFO = newton(th0, fcn, **optim_param,fit_param=fit_param)
            else:
                raise(NameError('not implemented yet'))

            # Record results
            T.loc[s,('likelihood',m_names[i])] = l
            T.loc[s,('iterations',m_names[i])] = INFO['iter']+1
            T.loc[s,('noise',m_names[i])] = exp(th[-Noise.n_param])
            if fit_scale:
                T.loc[s,('scale',m_names[i])] = exp(th[m.n_param])

            # Record theta parameters
            if theta[i] is None:
                theta[i] = np.zeros((th.shape[0],n_subj))
            theta[i][:,s] = th

            # If requested, return the second derivative of the likelihood
            if return_second_deriv:
                if dLdhh[i] is None:
                    dLdhh[i] = np.zeros((n_subj,th.shape[0],th.shape[0]))
                l,dl,dLdhh[i][s,:,:] = fcn(th)
                # log(det(inv(dLdhh))) = 2*sum(log(diag(cholesky(dLdhh))))
                if log(cond(dLdhh[i][s,:,:]))>16:
                    T.loc[s,('logdetPosterior',m_names[i])] = np.nan
                else:
                    T.loc[s,('logdetPosterior',m_names[i])] = 2 * sum(log(diag(cholesky(dLdhh[i][s,:,:]))))
    if return_second_deriv:
        return T,theta,dLdhh
    else:
        return T,theta

def fit_model_group(Data, M, fixed_effect='block', fit_scale=False,
                    scale_prior = 1e3, noise_cov=None, algorithm=None,
                    optim_param={}, theta0=None, verbose=True,
                    return_second_deriv=False):
    """ Fits PCM models(s) to a group of subjects

    The model parameters are (by default) shared across subjects.
    Scale and noise parameters are individual for each subject.
    Some model parameters can also be made individual by setting M.common_param

    Parameters:
        Data (list of pcm.Datasets):
            List data set has partition and condition descriptors
        M (pcm.Model or list of pcm.Models):
            Models to be fitted on the data sets. Optional field M.common_param indicates which model parameters are common to the group (True) and which ones are fit individually (False)
        fixed effect:
            None, 'block', or nd-array / list of nd-arrays. Default ('block') add an intercept for each partition
        fit_scale (bool):
            Fit a additional scale parameter for each subject? Default is set to False.
        scale_prior (float):
            Prior variance for log-normal prior on scale parameter
        algorithm (string):
            Either 'newton' or 'minimize' - provides over-write for model specific algorithms
        noise_cov:
            None (i.i.d), 'block', or optional specific covariance structure of the noise
        optim_param (dict):
            Additional paramters to be passed to the optimizer
        theta0 (list of np.arrays):
            List of starting values (same format as return argument theta)
        verbose (bool):
            Provide printout of progress? Default: True

    Returns:
        T (pandas.dataframe):
            Dataframe with the fields:
            SN:                 Subject number
            likelihood:         log-likelihood
            scale:              Scale parameter (if fitscale = 1)-exp(theta_s)
            noise:              Noise parameter- exp(theta_eps)
            iterations:         Number of interations for model fit
            time:               Elapsed time in sec

        theta (list of np.arrays):
            List of estimated model parameters each one is a vector with
            #num_commonparams + #num_singleparams x #numSubj elements

        G_pred (list of np.arrays):
            List of estimated G-matrices under the model
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

   # Get model names & Common parameters
    m_names = []
    for m in M:
        m_names.append(m.name)
        if hasattr(m,'common_param'):
            m.common_param = np.array(m.common_param)
        else:
            m.common_param = np.ones((m.n_param,), dtype=np.bool_)

    # Preallocate output structures
    iterab = [['likelihood','noise','scale','iterations'],m_names]
    index = pd.MultiIndex.from_product(iterab, names=['variable', 'model'])
    T = pd.DataFrame(np.zeros((n_subj, n_model * 4)), columns=index)
    theta = [None] * n_model
    if return_second_deriv:
        dLdhh = [None] * n_model

    # Determine optimal algorithm for each of the models
    # M = pcm.optimize.best_algorithm(M,algorithm)

    # Prepare the data for all the subjects
    Z, X, YY, n_channel, Noise, G_hat = set_up_fit_group(Data,
            fixed_effect = fixed_effect, noise_cov = noise_cov)

    # Average second moment
    G_avrg = sum(G_hat, axis=0) / n_subj

    # Initialize the different indices
    indx_scale = [None] * n_subj
    indx_noise = [None] * n_subj

    for i,m in enumerate(M):
        if verbose:
            print('Fitting group model',i)
        # Use externally provided theta, if provided
        if (theta0 is not None) and (len(theta0) >= i-1):
            th0 = theta0[i]
        else:
            # Get starting guess for theta0 is not provided
            th_gr = m.get_theta0(G_avrg) # Group theta
            th0 = th_gr[m.common_param]
            for s in range(n_subj):
                th0 = np.concatenate((th0,th_gr[np.logical_not(m.common_param)]))
                if (fit_scale):
                    indx_scale[s]=th0.shape[0]
                    G0,_ = m.predict(th_gr)
                    scale0 = get_scale0(G0, G_hat[s])
                    th0 = np.concatenate((th0,scale0))
                indx_noise[s]=th0.shape[0]
                th0 = np.concatenate((th0,Noise[s].get_theta0(Data[s].measurements, Z, X)))

        #  Now do the fitting, using the preferred optimization routine
        if (m.algorithm=='newton'):
            fcn = lambda x: likelihood_group(x, m, YY, Z, X=X,
                Noise = Noise, fit_scale = fit_scale, scale_prior=scale_prior, return_deriv = 2,n_channel=n_channel)
            theta[i], l, INFO = newton(th0, fcn, **optim_param)
        else:
            raise(NameError('not implemented yet'))

        # If requested, return the second derivative of the likelihood
        if return_second_deriv:
            l,dl,dLdhh[i] = fcn(theta[i])

        res = likelihood_group(theta[i], m, YY, Z, X=X,
                Noise = Noise, fit_scale = fit_scale, scale_prior=scale_prior,return_deriv = 0,return_individ=True, n_channel=n_channel)
        T['likelihood',m_names[i]] = -res[0]
        T['iterations',m_names[i]] = INFO['iter']+1
        T['noise',m_names[i]] = exp(theta[i][indx_noise])
        if (fit_scale):
            T['scale',m_names[i]] = exp(theta[i][indx_scale])

    if return_second_deriv:
        return T,theta,dLdhh
    else:
        return T,theta

def fit_model_group_crossval(Data, M, fixed_effect='block', fit_scale=False,
                    scale_prior = 1e3, noise_cov=None, algorithm=None,
                    optim_param={}, theta0=None, verbose=True):
    """Fits PCM model(sto N-1 subjects and evaluates the likelihood on the Nth subject.

    Only the common model parameters are shared across subjects.The scale and noise parameters
    are still fitted to each subject. Some model parameters can also be made individual by setting M.common_param to False

    Parameters:
        Data (list of pcm.Datasets):
            List data set has partition and condition descriptors
        M (pcm.Model or list of pcm.Models):
            Models to be fitted on the data sets. Optional field M.common_param indicates which model parameters are common to the group (True) and which ones are fit individually (False)
        fixed effect:
            None, 'block', or nd-array. Default ('block') add an intercept for each partition
        fit_scale (bool):
            Fit a additional scale parameter for each subject? Default is set to False.
        scale_prior (float):
            Prior variance for log-normal prior on scale parameter
        algorithm (string):
            Either 'newton' or 'minimize' - provides over-write for model specific algorithms
        noise_cov:
            None (i.i.d), 'block', or optional specific covariance structure of the noise
        optim_param (dict):
            Additional paramters to be passed to the optimizer
        theta0 (list of np.arrays):
            List of starting values (same format as return argument theta)
        verbose (bool):
            Provide printout of progress? Default: True

    Returns:
        T (pandas.dataframe):
            Dataframe with the fields:
            SN:                 Subject number
            likelihood:         log-likelihood
            scale:              Scale parameter (if fitscale = 1)-exp(theta_s)
            noise:              Noise parameter- exp(theta_eps)
            iterations:         Number of interations for model fit
            time:               Elapsed time in sec

        theta (list of np.arrays):
            List of estimated model parameters - common group parameters come from the training data, individual parameters from the testing data

        G_pred (list of np.arrays):
            List of estimated G-matrices under the model

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
    iterab = [['likelihood','noise','scale'],m_names]
    index = pd.MultiIndex.from_product(iterab, names=['variable', 'model'])
    T = pd.DataFrame(np.zeros((n_subj, n_model * 3)), columns=index)
    theta_loo = [None] * n_model
    theta_ind = [None] * n_model

    # Intialize data for groupfit
    Z, X, YY, n_channel, Noise, G_hat = set_up_fit_group(Data, fixed_effect = fixed_effect, noise_cov = noise_cov)

    # Get starting values as for a group fit
    G_avrg = sum(G_hat, axis=0) / n_subj
    for i,m in enumerate(M):
        if verbose:
            print('Fitting group cross model',i)
        # Use externally provided theta, if provided
        if (theta0 is not None) and (len(theta0) >= i-1):
            th0 = theta0[i]
        else:
            th_gr = m.get_theta0(G_avrg)
            th0 = th_gr[m.common_param]
            # Keep track which one belongs to which subject (-1 is group)
            param_indx = np.ones((m.common_param.sum(),))*-1

            # Get starting guess for theta0 is not provided
            for s in range(n_subj):
                ths = th_gr[np.logical_not(m.common_param)]
                if (fit_scale):
                    G0,_ = m.predict(th_gr)
                    scale0 = get_scale0(G0, G_hat[s])
                    ths = np.concatenate((ths,scale0))
                ths = np.concatenate((ths,Noise[s].get_theta0(Data[s].measurements, Z[s], X[s])))
                # add to the group parameters and list
                th0 = np.concatenate((th0,ths))
                param_indx = np.concatenate((param_indx,np.full((len(ths),),s)))


        #  Get a group fit
        if (m.algorithm=='newton'):
            fcn = lambda x: likelihood_group(x, m, YY, Z, X=X,
                Noise = Noise, fit_scale = fit_scale, scale_prior=scale_prior, return_deriv = 2,n_channel=n_channel)
            theta_gr, l, INFO = newton(th0, fcn, **optim_param)
        else:
            raise(NameError('not implemented yet'))

        theta_ind[i],indx = group_to_individ_param(theta_gr,m,n_subj)

        # Loop over subjects can fit the rest to get group parameters
        for s in range(n_subj):
            notS = np.arange(n_subj) != s # Get indices of training group
            pNotS = param_indx != s

            # Use the group fit parameters, as starting
            th0 = theta_gr[pNotS]

            #  Now do the fitting, using the preferred optimization routine
            if (m.algorithm=='newton'):
                fcn = lambda x: likelihood_group(x, m, YY[notS], Z[notS],
                                            X=X[notS], Noise = Noise[notS], fit_scale = fit_scale, scale_prior = scale_prior, return_deriv = 2, n_channel=n_channel[notS])
                theta_loo, l, INFO = newton(th0, fcn, **optim_param)
            else:
                raise(NameError('not implemented yet'))

            # Evaluate likelihood on the left-out subject
            th0 = theta_ind[i][:,s]    # Get subject based parameter from group
            fit_param = param_indx[indx[:,s]]!=-1  # Get the [parameters that need to be fitted
            # Set the non-fitted parameters to the values from N-1 subjects
            not_fit = np.logical_not(fit_param)
            th0[not_fit]=theta_loo[param_indx[pNotS]==-1]
            fcn = lambda x: likelihood_individ(x, m, YY[s], Z[s], X=X[s],
                    Noise = Noise[s], n_channel=n_channel[s],
                    fit_scale = fit_scale, scale_prior = scale_prior,
                    return_deriv = 2)
            theta_ind[i][:,s], l, INF2 = newton(th0, fcn, **optim_param,fit_param=fit_param)
            # record results into the array
            T['likelihood',m_names[i]][s] = l
            if (fit_scale):
                T['scale',m_names[i]][s] = exp(theta_ind[i][-2,s])
            T['noise',m_names[i]][s] = exp(theta_ind[i][-1,s])
    return T,theta_ind


def sample_model_individ(Data, M,
                    fixed_effect='block',
                    fit_scale=False,
                    scale_prior = 1e3,
                    noise_cov=None,
                    n_mcmc_samples=10000,
                    n_local_samples= 2000):
    """ Approximates the posterior of the parameters of a group model using MCMC sampling
    If requested, it also tries to approximated the marginal likelihood using mnethod outlined in Chib & Jeliazkov (2011)

    Parameters:
        Data (list of pcm.Datasets):
            List data set has partition and condition descriptors
        M (pcm.Model):
            Models to sampled
        fixed effect:
            None, 'block', or nd-array / list of nd-arrays. Default ('block') add an intercept for each partition
        fit_scale (bool):
            Fit a additional scale parameter for each subject? Default is set to False.
        scale_prior (float):
            Prior variance for log-normal prior on scale parameter
        noise_cov:
            None (i.i.d), 'block', or optional specific covariance structure of the noise
        sample_param (dict):
            Additional paramters to be passed to MCMC sampler
        theta0 (np.array):
            starting values

    Returns:
        theta (np.array):
            Sampled parameters
        l (np.array):
            Log-likelihood corresponding to the sampled parameters
    """
    # Get an initial fit to the data
    T,th_fit,dLL = fit_model_individ(Data, M,
                    fixed_effect=fixed_effect,
                    fit_scale=fit_scale,
                    scale_prior = 1e3,
                    noise_cov=noise_cov,
                    return_second_deriv=True)
    th0 = th_fit[0].squeeze()
    n_param = th0.shape[0]

    # Prepare the data for all the subjects
    Z, X, YY, n_channel, Noise, G_hat = set_up_fit(Data,
            fixed_effect = fixed_effect, noise_cov = noise_cov)

    #  Now do the fitting, using the preferred optimization routine
    fcn = lambda x: likelihood_individ(x, M, YY, Z, X=X,
            Noise = Noise, fit_scale = fit_scale, scale_prior=scale_prior, return_deriv = 0,n_channel=n_channel)
    proposal_sd = 1/np.sqrt(np.diag(dLL[0][0]))
    proposal_sd[proposal_sd>5]=5
    proposal_sd[proposal_sd<0.001]=0.001

    theta, l  = mcmc(th0, fcn,
                     n_samples = n_mcmc_samples,
                     proposal_sd = proposal_sd)

    # Get samples around the maximum likelihood
    l_max = -fcn(th0)[0]
    l_local = np.zeros((n_local_samples,))
    z = np.random.normal(0,1,(n_param,n_local_samples))
    th_local = z * proposal_sd.reshape(-1,1) + th0.reshape(-1,1)
    for i in range(n_local_samples):
        l_local[i] = -fcn(th_local[:,i])[0]

    # Numerator of the marginal likelihood
    logq = -n_param/2 * np.log(2*np.pi) -0.5 * np.log(proposal_sd**2).sum() * -0.5*np.sum((theta-th0.reshape(-1,1))**2/proposal_sd.reshape(-1,1)**2,axis=0)
    alp = np.minimum(0,l-l_max)
    num = (np.exp(alp+logq)).mean()

    # Denominator of the marginal likelihood
    alp = np.minimum(0,l_local-l_max)
    den = (np.exp(alp)).mean()
    log_posterior = np.log(num/den)     # Log-posterior evaluated at the maximum likelihood

    # Log marginal likelihood
    log_marg_lik = l_max - log_posterior

    return theta,l,log_marg_lik


def sample_model_group(Data, M,
                    fixed_effect='block',
                    fit_scale=False,
                    scale_prior = 1e3,
                    noise_cov=None,
                    sample_param={'n_samples':10000,'burn_in':100},
                    theta0=None,
                    verbose = True,
                    proposal_sd = None):
    """ Approximates the posterior of the parameters of a group model using MCMC sampling

    The model parameters are (by default) shared across subjects.
    Scale and noise parameters are individual for each subject.
    Some model parameters can also be made individual by setting M.common_param

    Parameters:
        Data (list of pcm.Datasets):
            List data set has partition and condition descriptors
        M (pcm.Model):
            Models to sampled
        fixed effect:
            None, 'block', or nd-array / list of nd-arrays. Default ('block') add an intercept for each partition
        fit_scale (bool):
            Fit a additional scale parameter for each subject? Default is set to False.
        scale_prior (float):
            Prior variance for log-normal prior on scale parameter
        noise_cov:
            None (i.i.d), 'block', or optional specific covariance structure of the noise
        sample_param (dict):
            Additional paramters to be passed to MCMC sampler
        theta0 (np.array):
            starting values

    Returns:
        theta (np.array):
            Sampled parameters
        l (np.array):
            Log-likelihood corresponding to the sampled parameters
        G_pred (list of np.arrays):
            List of estimated G-matrices under the model
    """
    # Get the number of subjects
    if type(Data) is list:
        n_subj = len(Data)
    else:
        n_subj = 1
        Data = [Data]

   # Get Common parameters
    if hasattr(M,'common_param'):
        M.common_param = np.array(M.common_param)
    else:
        M.common_param = np.ones((M.n_param,), dtype=np.bool_)

    # Prepare the data for all the subjects
    Z, X, YY, n_channel, Noise, G_hat = set_up_fit_group(Data,
            fixed_effect = fixed_effect, noise_cov = noise_cov)

    # Average second moment
    G_avrg = sum(G_hat, axis=0) / n_subj

    # Initialize the different indices
    indx_scale = [None] * n_subj
    indx_noise = [None] * n_subj

    if verbose:
        print('Sampling group model')

    # Use externally provided theta, if provided
    if (theta0 is not None):
        th0 = theta0
    else:   # Get starting guess for theta0 is not provided
        th_gr = M.get_theta0(G_avrg)
        th0 = th_gr[M.common_param]
        for s in range(n_subj):
            th0 = np.concatenate((th0,th_gr[np.logical_not(M.common_param)]))
            if (fit_scale):
                indx_scale[s]=th0.shape[0]
                G0,_ = M.predict(th_gr)
                scale0 = get_scale0(G0, G_hat[s])
                th0 = np.concatenate((th0,scale0))
            indx_noise[s]=th0.shape[0]
            th0 = np.concatenate((th0,Noise[s].get_theta0(Data[s].measurements, Z[s], X[s])))


    #  Now do the fitting, using the preferred optimization routine
    fcn = lambda x: likelihood_group(x, M, YY, Z, X=X,
            Noise = Noise, fit_scale = fit_scale, scale_prior=scale_prior, return_deriv = 0,n_channel=n_channel)
    theta, l  = mcmc(th0, fcn, **sample_param, proposal_sd = proposal_sd)
    return theta,l


def set_up_fit(Data, fixed_effect = 'block', noise_cov = None):
    """Utility routine pre-calculates and sets design matrices, etc for the PCM fit

    Parameters:
        Data (pcm.dataset):
            Contains activity data (measurement), and obs_descriptors partition and condition
        fixed_effect:
            Can be None, 'block', or a design matrix. 'block' includes an intercept for each partition.
        noise_cov:
            Can be None: (i.i.d noise), 'block': a common noise paramter or a List of noise covariances for the different partitions

    Returns:
        Z (np.array):
            Design matrix for random effects
        X (np.array):
            Design matrix for fixed effects
        YY (np.array):
            Quadratic form of the data (Y Y')
        Noise (pcm.model.NoiseModel):
            Noise model
        G_hat (np.array):
            Crossvalidated estimate of second moment of U

    """

    # Make design matrix
    cV = Data.obs_descriptors['cond_vec']
    if cV.ndim == 1:
        Z = pcm.matrix.indicator(cV)
    elif cV.ndim == 2:
        Z = cV
    n_reg = Z.shape[1]

    # Get data
    Y = Data.measurements
    if np.iscomplexobj(Y):
        raise(NameError('Data array contains complex numbers: PCM only works for real data.'))
    N, n_channel = Y.shape
    YY = Y @ Y.T

    # Initialize fixed effects
    part_vec = Data.obs_descriptors['part_vec']
    if fixed_effect is None:
        X = None
    elif fixed_effect=='block':
        X = pcm.matrix.indicator(part_vec)
    else:
        X = fixed_effect

    # Now choose the noise model
    if noise_cov is None:
        Noise = IndependentNoise()
    elif noise_cov == 'block':
        Noise = BlockPlusIndepNoise(part_vec)
    else:
        raise(NameError('Arbitrary covariance matrices are not yet implemented'))

    # Get a cross-validated estimate of G
    G_hat, _ = pcm.util.est_G_crossval(Y, Z, part_vec, X = X)

    return [Z, X, YY, n_channel, Noise, G_hat]

def set_up_fit_group(Data, fixed_effect = 'block', noise_cov = None):
    """Pre-calculates and sets design matrices, etc for the PCM fit for a full group

    Parameters:
        Data (list of pcm.dataset):
            Contains activity data (measurement), and obs_descriptors partition and condition
        fixed_effect:
            Can be None, 'block', or a design matrix. 'block' includes an intercept for each partition.
        noise_cov:
            Can be None: (i.i.d noise), 'block': a common noise paramter or a List of noise covariances for the different partitions

    Returns:
        Z (np.array): Design matrix for random effects
        X (np.array): Design matrix for fixed effects
        YY (np.array): Quadratic form of the data (Y Y')
        Noise (NoiseModel): Noise model
        G_hat (np.array): Crossvalidated estimate of second moment of U

    """
    n_subj = len(Data)
    Z = np.empty((n_subj,),dtype=object)
    X = np.empty((n_subj,),dtype=object)
    YY = np.empty((n_subj,),dtype=object)
    n_channel = np.zeros((n_subj,),dtype=int)
    G_hat = np.empty((n_subj,),dtype=object)
    Noise = np.empty((n_subj,),dtype=object)
    for s in range(n_subj):
        if type(fixed_effect) is list:
            fe = fixed_effect[s]
        else:
            fe = fixed_effect
        if type(noise_cov) is list:
            nc = noise_cov[s]
        else:
            nc = noise_cov
        Z[s], X[s], YY[s], n_channel[s], Noise[s], G_hat[s] = pcm.inference.set_up_fit(Data[s],
                    fixed_effect = fe, noise_cov = nc)
    return Z, X ,YY, n_channel, Noise, G_hat

def get_scale0(G,G_hat):
    """"
    Get approximate (log-)scaling parameter between predicted G and estimated G_hat

    Parameters:
        G (numpy.ndarray0)
            Predicted G matrix by the model
        G_hat (numpy.ndarry0)
            Directly estimated G from the data
    Returns:
        scale0:
            log-scaling parameter
    """
    g = G.reshape((-1,))
    g_hat = G_hat.reshape((-1,))
    scaling = (g @ g_hat) / (g @ g + 10e-5)
    if (scaling < 10e-5):
        scaling = 10e-5
    return log(np.array([scaling]))
