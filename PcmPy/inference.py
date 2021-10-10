"""
Inference module for PCM toolbox with main functionality for model fitting and evaluation.
@author: jdiedrichsen
"""

import numpy as np
from numpy.linalg import solve, eigh, cholesky
from numpy import sum, diag, log, eye, exp, trace, einsum
import pandas as pd
import PcmPy as pcm
from PcmPy.model import IndependentNoise, BlockPlusIndepNoise
from PcmPy.optimize import newton


def likelihood_individ(theta, M, YY, Z, X=None,
                       Noise = IndependentNoise(),
                       n_channel=1, fit_scale=False, scale_prior = 1e3, return_deriv=0):
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
            0: Only return negative loglikelihood
            1: Return first derivative
            2: Return first and second derivative (default)

    Returns:
        negloglike (double):
            Negative log-likelihood of the data under a model

        dLdtheta (1d-np.array):
            First derivative of negloglike in respect to the parameters

        ddLdtheta2 (2d-np.array):
            Second derivative of negloglike in respect to the parameters

    """

    N = YY.shape[0]
    Q = Z.shape[1]
    n_param = theta.shape[0]

    # Get G-matrix and derivative of G-matrix in respect to parameters
    model_params = theta[range(M.n_param)]
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
    if fit_scale:
        llik -= scale_param**2 / (2 * scale_prior) # Add prior

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
            Vector of (log-)model parameters consisting of common model parameters (M.n_param or sum of M.common_param),
            participant-specific parameters (interated by subject), unique model parameters (not in common_param),
            scale parameter,noise parameters
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

    # Determine the common parameters to the group
    if hasattr(M,'common_param'):
        common_param = M.common_param
    else:
        common_param = np.ones((M.n_param,),dtype=np.bool_)

    # Get the number of parameters
    n_common = np.sum(common_param) # Number of common params
    n_modsu = M.n_param - n_common # Number of subject-specific model params
    n_scale = int(fit_scale) # Number of scale parameters
    n_noise = Noise[0].n_param # Number of noise params
    n_per_subj = n_modsu + n_scale + n_noise # Number of parameters per subj

    # Generate the indices into the theta vector
    indx_common = np.array(range(n_common))
    indx_subj = np.arange(n_common, n_common + n_subj * n_per_subj, n_per_subj, dtype = int)
    indx_subj = indx_subj.reshape((1,-1))
    indx_modsu = np.zeros((n_modsu,1),dtype = int) + indx_subj
    indx_scale = np.zeros((n_scale,1),dtype = int) + n_modsu + indx_subj
    indx_noise = np.array(range(n_noise),dtype = int).T + n_scale + n_modsu + indx_subj

    # preallocate the arrays
    nl = np.zeros((n_subj,))
    dFdh = np.zeros((n_subj,n_param))
    dFdhh = np.zeros((n_subj,n_param,n_param))

    # Loop over subjects and get individual likelihoods
    for s in range(n_subj):
        # Pick out the correct places for the group parameter vector for each subj
        indx_model = np.zeros((M.n_param,), dtype=int)
        indx_model[common_param]=indx_common
        indx_model[np.logical_not(common_param)]=indx_modsu[:,s]
        indx = np.concatenate([indx_model, indx_scale[:,s], indx_noise[:,s]])
        ths = theta[indx]
        # Get individual likelihood
        res = likelihood_individ(ths, M, YY[s], Z[s], X[s],
                       Noise[s], n_channel[s], fit_scale = fit_scale, scale_prior = scale_prior, return_deriv = return_deriv)
        iS = indx_scale[0,s]
        nl[s] = res[0]
        if return_deriv>0:
            dFdh[s, indx] = res[1]
        if return_deriv==2:
            ixgrid = np.ix_([s],indx,indx)
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

def fit_model_individ(Data, M, fixed_effect='block', fit_scale=False,
                    scale_prior = 1e3, noise_cov=None, algorithm=None,
                    optim_param={}, theta0=None, verbose = True):
    """Fits Models to a data set inidividually.

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
    if type(M) is list:
        n_model = len(M)
    else:
        n_model = 1
        M = [M]

 # Get model names
    m_names = []
    for m in range(n_model):
        m_names.append(M[m].name)

    # Preallocate output structures
    iterab = [['likelihood','noise','iterations'],m_names]
    index = pd.MultiIndex.from_product(iterab, names=['variable', 'model'])
    T = pd.DataFrame(np.zeros((n_subj, n_model * 3)), columns=index)
    theta = [None] * n_model

    # Determine optimal algorithm for each of the models
    # M = pcm.optimize.best_algorithm(M,algorithm)

    # Loop over subject and models and provide inidivdual fits
    for s in range(n_subj):
        Z,X,YY,n_channel,Noise,G_hat = set_up_fit(Data[s],
                                                fixed_effect = fixed_effect,
                                                noise_cov = noise_cov)
        for m in range(n_model):
            if verbose:
                print('Fitting Subj',s,'model',m)
            # Get starting guess for theta0 is not provideddf
            if (theta0 is None) or (len(theta0) <= m) or (theta0[m].shape[1]<s):
                M[m].set_theta0(G_hat)
                th0 = M[m].theta0
                if (fit_scale):
                    G_pred, _ = M[m].predict(M[m].theta0)
                    scale0 = get_scale0(G_pred, G_hat)
                    th0 = np.concatenate((th0,scale0))
                th0 = np.concatenate((th0,Noise.theta0))
            else:
                th0 = theta0[m][:,s]

            #  Now do the fitting, using the preferred optimization routine
            if (M[m].algorithm=='newton'):
                fcn = lambda x: likelihood_individ(x, M[m], YY, Z, X=X,
                                Noise = Noise, fit_scale = fit_scale, scale_prior = scale_prior, return_deriv = 2,n_channel=n_channel)
                th, l, INFO = newton(th0, fcn, **optim_param)
            else:
                raise(NameError('not implemented yet'))

            if theta[m] is None:
                theta[m] = np.zeros((th.shape[0],n_subj))
            theta[m][:,s] = th
            T.loc[s,('likelihood',m_names[m])] = l
            T.loc[s,('iterations',m_names[m])] = INFO['iter']+1
            T.loc[s,('noise',m_names[m])] = exp(th[-Noise.n_param])
            if fit_scale:
                T.loc[s,('scale',m_names[m])] = exp(th[M[m].n_param])
    return [T,theta]

def fit_model_group(Data, M, fixed_effect='block', fit_scale=False,
                    scale_prior = 1e3, noise_cov=None, algorithm=None,
                    optim_param={}, theta0=None, verbose=True):
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
            List of estimated model parameters, each a

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
    if type(M) is list:
        n_model = len(M)
    else:
        n_model = 1
        M = [M]

   # Get model names
    m_names = []
    for m in range(n_model):
        m_names.append(M[m].name)

    # Preallocate output structures
    iterab = [['likelihood','noise','scale','iterations'],m_names]
    index = pd.MultiIndex.from_product(iterab, names=['variable', 'model'])
    T = pd.DataFrame(np.zeros((n_subj, n_model * 4)), columns=index)
    theta = [None] * n_model

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

    for m in range(n_model):
        if verbose:
            print('Fitting model',m)
        # Get starting guess for theta0 is not provided
        if hasattr(M[m],'common_param'):
            common = M[m].common_param
        else:
            common = np.ones((M[m].n_param,), dtype=np.bool_)
        M[m].set_theta0(G_avrg)
        th0 = M[m].theta0[common]
        for s in range(n_subj):
            th0 = np.concatenate((th0,M[m].theta0[np.logical_not(common)]))
            if (fit_scale):
                indx_scale[s]=th0.shape[0]
                G0,_ = M[m].predict(M[m].theta0)
                scale0 = get_scale0(G0, G_hat[s])
                th0 = np.concatenate((th0,scale0))
            indx_noise[s]=th0.shape[0]
            th0 = np.concatenate((th0,Noise[s].theta0))
        if (theta0 is not None) and (len(theta0) >= m-1):
            th0 = theta0[m]

        #  Now do the fitting, using the preferred optimization routine
        if (M[m].algorithm=='newton'):
            fcn = lambda x: likelihood_group(x, M[m], YY, Z, X=X,
                Noise = Noise, fit_scale = fit_scale, scale_prior=scale_prior, return_deriv = 2,n_channel=n_channel)
            theta[m], l, INFO = newton(th0, fcn, **optim_param)
        else:
            raise(NameError('not implemented yet'))

        res = likelihood_group(theta[m], M[m], YY, Z, X=X,
                Noise = Noise, fit_scale = fit_scale, scale_prior=scale_prior,return_deriv = 0,return_individ=True, n_channel=n_channel)
        T['likelihood',m_names[m]] = -res[0]
        T['iterations',m_names[m]] = INFO['iter']+1
        T['noise',m_names[m]] = exp(theta[m][indx_noise])
        if (fit_scale):
            T['scale',m_names[m]] = exp(theta[m][indx_scale])
    return [T,theta]

def fit_model_group_crossval(Data, M, fixed_effect='block', fit_scale=False,
                    scale_prior = 1e3, noise_cov=None, algorithm=None,
                    optim_param={}, theta0=None, verbose=True):
    """Fits PCM model(sto N-1 subjects and evaluates the likelihood on the Nth subject.

    Only the common model parameters are shared across subjects.The scale and noise parameters
    are still fitted to each subject. Some model parameters can also be made individual by setting M.common_param

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
            List of estimated model parameters - common group parameters come from the training data, scale and noise parameter from the testing data

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
    if type(M) is list:
        n_model = len(M)
    else:
        n_model = 1
        M = [M]

    # Get model names
    m_names = []
    for m in range(n_model):
        m_names.append(M[m].name)

    # Preallocate output structures
    iterab = [['likelihood','noise','scale'],m_names]
    index = pd.MultiIndex.from_product(iterab, names=['variable', 'model'])
    T = pd.DataFrame(np.zeros((n_subj, n_model * 3)), columns=index)
    theta = [None] * n_model

    # Determine optimal algorithm for each of the models
    # M = pcm.optimize.best_algorithm(M,algorithm)
    Z, X, YY, n_channel, Noise, G_hat = set_up_fit_group(Data, fixed_effect = fixed_effect, noise_cov = None)

    # Get starting values as for a group fit
    G_avrg = sum(G_hat, axis=0) / n_subj
    for m in range(n_model):
        if verbose:
            print('Fitting model',m)

        # Get starting guess for theta0 is not provided
        if hasattr(M[m],'common_param'):
            common = M[m].common_param
        else:
            common = np.ones((M[m].n_param,), dtype=np.bool_)
        not_common = np.logical_not(common)
        n_modsu = np.sum(not_common) # Number of subject-specific parameters
        M[m].set_theta0(G_avrg)
        th0 = M[m].theta0[common]

        # Keep track of what subject the parameter belongs to
        param_indx = np.ones((np.sum(common),)) * -1
        for s in range(n_subj):
            th0 = np.concatenate((th0,M[m].theta0[not_common]))
            param_indx = np.concatenate((param_indx,s * np.ones((n_modsu,))))

            if (fit_scale):
                G0,_ = M[m].predict(M[m].theta0)
                scale0 = get_scale0(G0, G_hat[s])
                th0 = np.concatenate((th0,scale0))
                param_indx = np.concatenate((param_indx,np.ones((1,)) * s))

            th0 = np.concatenate((th0,Noise[s].theta0))
            param_indx = np.concatenate((param_indx, s * np.ones((Noise[s].n_param,))))
            if (theta0 is not None) and (len(theta0) >= m-1):
                th0 = theta0[m]

        # Initialize parameter array for group
        theta[m] = np.zeros((th0.shape[0],n_subj))

        # Loop over subjects can fit the rest to get group parameters
        for s in range(n_subj):
            notS = np.arange(n_subj) != s # Get indices of training group
            pNotS = param_indx != s
            # Set theta0 and model (for direct estimation)
            G_avrg = sum(G_hat[notS], axis=0) / n_subj
            M[m].set_theta0(G_avrg)

            #  Now do the fitting, using the preferred optimization routine
            if (M[m].algorithm=='newton'):
                fcn = lambda x: likelihood_group(x, M[m], YY[notS], Z[notS],
                                            X=X[notS], Noise = Noise[notS], fit_scale = fit_scale, scale_prior = scale_prior, return_deriv = 2, n_channel=n_channel[notS])
                theta[m][:,s], l, INFO = newton(th0, fcn, **optim_param)
            else:
                raise(NameError('not implemented yet'))

            # Evaluate likelihood on the left-out subject
            if hasattr(M[m],'common_param'):
                raise(NameError('Group crossval with subject specific params not implemented yet'))
            else:
                thm = theta[m][param_indx==-1, s] # Common model parameter
                G_group, _ = M[m].predict(thm) # Predicted second moment matrix
                Mindiv = pcm.model.FixedModel('name',G_group) # Make a fixed model
                p = param_indx == s # Parameters for this subject
                fcn = lambda x: likelihood_individ(x, Mindiv, YY[s], Z[s], X=X[s], Noise = Noise[s], n_channel=n_channel[s], fit_scale = fit_scale, scale_prior = scale_prior, return_deriv = 2)
                thi, l, INF2 = newton(th0[p], fcn, **optim_param)
            # record results into the array
            T['likelihood',m_names[m]][s] = l
            if (fit_scale):
                T['scale',m_names[m]][s] = exp(thi[0])
            T['noise',m_names[m]][s] = exp(thi[int(fit_scale)])
    return [T,theta]

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

    # Estimate noise parameters starting values
    Noise.set_theta0(Y,Z,X)
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
    scaling = (g @ g_hat) / (g @ g)
    if (scaling < 10e-5):
        scaling = 10e-5
    return log(np.array([scaling]))
