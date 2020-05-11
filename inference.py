"""
Inference module for PCM toolbox with main functionality for model fitting and evaluation.
    likelihood_individ: Likelihood for an individual data set
    likelihood_group: Likelihood with shared model parameters across group

@author: jdiedrichsen
"""

import numpy as np
from numpy.linalg import solve, eigh, cholesky
from numpy import sum, diag, log, eye, exp, trace, einsum
from PcmPy import model


def likelihood_individ(theta, M, YY, Z, X=None, 
                       Noise=model.IndependentNoise(),
                       n_channel=1, fit_scale=False, return_deriv=0):
    """
    Negative Log-Likelihood of the data and derivative in respect to the parameters

    Parameters:
        theta (np.array)
            Vector of (log-)model parameters: These include model, signal and noise parameters
        M (pcm.Model)
            Model object with predict function
        YY (2d-np.array)
            NxN Matrix of outer product of the activity data (Y*Y')
        Z (2d-np.array)
            NxQ Design matrix - relating the trials (N) to the random effects (Q)
        X (np.array)
            Fixed effects design matrix - will be accounted for by ReML
        Noise (pcm.Noisemodel)
            Pcm-noise mode to model block-effects (default: Indentity)
        n_channel (int)
            Number of channels
        fit_scale (bool)
            Fit a scaling parameter for the model (default is False)
        return_deriv (int)
            0: Do not return any derivative
            1: Return first derivative
            2: Return first and second derivative (default)
    
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
    idx = Glambda > (10e-10) # Increased to 10*eps from 2*eps
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
        llik = llik - n_channel * sum(log(diag(cholesky(X.T @ iV @X)))) #

    # If no derivative - exit here 
    if return_deriv == 0: 
        return -llik

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
        dLdtheta[i] = -n_channel / 2 * trace(iVdV[i]) + 0.5 * einsum('ij,ij->',iVdV[i], B)

    # If only first derivative, exit here 
    if return_deriv == 1: 
        return -llik, -dLdtheta

    # Calculate expected second derivative
    d2L = np.zeros((n_param,n_param))
    for i in range(n_param):
        for j in range(i, n_param):
            d2L[i, j] = -n_channel / 2 * einsum('ij,ij->',iVdV[i],iVdV[j])
            d2L[j, i] = d2L[i, j]
    if return_deriv == 2: 
        return -llik, -dLdtheta, -d2L
    else: 
        raise NameError('return_deriv needs to be 0, 1 or 2')

def fit_model_individ(Data, M, condition_vec=None, partition_vec=None,run_effect='fixed', fit_scale=False, noise_cov=None, algorithm=None, optim_param=None,theta0=None):
    """
    Fits pattern component model(s) specified by M to data from a number of
    subjects.
    The model parameters are all individually fit.
    INPUT:
        Data (pcm.Dataset or list of pcm.Datasets)
            List data set has partition and condition descriptors
        M (pcm.Model or list of pcm.Models)
            Models to be fitted on the data sets
        condition_vec (np.array or list of np.arrays)
            condition assignment vector
            for each subject. Rows of conditionVec{subj} define
            condition assignment of rows of Y{subj}.
            If a single vector is provided, it is assumed to me the
            same for all subjects.
            If the (elements of) conditionVec are matrices, it is
            assumed to be the design matrix Z, allowing the
            specification individualized models
        partition_vec (np.array or list of np.arrays)
            partition assignment vector
            for each subject. 
        run_effect (string)
            'random': Models variance of the run effect for each subject
                as a seperate random effects parameter.
            'fixed': Consider run effect a fixed effect, will be removed
                 implicitly using ReML.
            'none': No modeling of the run Effect
        fit_scale (bool)
            Fit a additional scale parameter for each subject? Default is set to False. 
        algorithm (string)
            Either 'newton' or 'minimize' - provides over-write for model specific algorithms
        noise_cov
            Optional specific covariance structure of the noise
            {#Subjects} = cell array of N_s x N_s matrices 
            Number of max minimization iterations. Default is 1000.
            S(#Subjects).S and .invS: Structure of the N_s x N_s 
            normal and inverse covariances matrices
        optim_param (dict) 
            Additional paramters to be passed to the optimizer 
        theta0 (list of np.arrays)
            List of starting values (same format as return argument theta)
    Returns
        RES (pandas.dataframe)
            Dataframe with the fields: 
            SN:                 Subject number
            likelihood:         log-likelihood
            scale:              Scale parameter (if fitscale = 1)-exp(theta_s)
            noise:              Noise parameter- exp(theta_eps)
            run:                Run parameter (if run = 'random')
            iterations:         Number of interations for model fit
            time:               Elapsed time in sec
        theta (list of np.arrays) 
            List of estimated model parameters, each a
            #params x #numSubj np.array
        G_pred (list of np.arrays)
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
    
    # Preallocate output structures 
    iterab = [['likelihood','noise'],range(n_model)]
    index = pd.MultiIndex.from_product(iterab, names=['variable', 'model'])   
    RES = pd.Dataframe(np.zeros((n_subj, n_model*2)), columns=index)
    theta0 = [np.zeros((3,3))]*3

    # Determine optimal algorithm for each of the models
    # M = pcm.optimize.best_algorithm(M,algorithm)

    # Set up all parameters for the upcoming fit
    for s in range(n_subj):
        Z,X,YY,Noise,N,P,G_hat = set_up_fit(Data,partitionVec,conditionVec,runEffect,noise_cov = noise_cov)

    # Initialize list and dataframe for results

    # Loop over subject and models and provide inidivdual fits
    for s in range(n_subj): 
        for m in range (n_models):
            printf('Fitting Subj',s,'model',m)
            # Get starting guess for theta0 is not provideddf
            if len(theta0)<m or theta0[m].shape[1]<s:
                M[m].get_startingval(G[s,:,:])
                th0 = M[m].theta0
                if (fitScale):
                    G0 = predict(M[m],M[m].theta0)
                    g0 = G0.reshape((-1,))
                    g_hat = G_hat[s,:,:].reshape((-1,))
                    scaling = (g0 @ g_hat) / (g0 @ g0)
                    if (scaling < 10e-5):
                        scaling = 10e-5
                    th0 = np.concatinate(th0,log(scaling))
                th0 = np.concatinate(th0,Noise.theta0);
        
            #  Now do the fitting, using the preferred optimization routine
            if (M[m].fitAlgorithm=='newton'):
                fcn = @(x) pcm_likelihoodIndivid(x,YY{s},M{m},Z{s},X{s},P(s),OPT);
                [theta_hat{m}(:,s),T.likelihood(s,m),T.iterations(s,m),T.reg(s,m),INFO.regH{s,m},INFO.thetaH{s,m}]= ... 
                    pcm_NR(theta0{m}(:,s),fcn,fitOPT); 
            else:
                raise(NameError('not implemented yet')

            G_pred[m](:,:,s)  =  pcm_calculateG(M{m},theta_hat{m}(1:M{m}.numGparams,s));
            T.noise(s,m)      =  exp(theta_hat{m}(M{m}.numGparams+1,s));
            if (fitScale)
                T.scale(s,m)      = exp(theta_hat{m}(M{m}.numGparams+2,s));
            end
        
            T.time(s,m)       = toc;

def set_up_fit(Data, run_effect = 'none', noise_cov = None): 
    """ 
    Pre-calculates and sets design matrices, etc for the PCM fit 
    INPUT
        Data (pcm.dataset)
            Contains activity data (measurement), and obs_descriptors partition and condition
        run_effect
            For fmri data can be 'none', 'random', or 'fixed'
        noise_cov
            List of noise covariances for the different partitions 
    RETURNS 
        Z
            Design matrix for random effects
        X
            Design matrix for fixed effects 
        YY
            Quadratic form of the data (Y Y')
        Noise
            Noise model
        G_hat
            Crossvalidated estimate of second moment of U
    """
    # Make design matrix 
    cV = Data.obs_descriptors['condition']
    if cV.ndim == 1:
        Z = pcm.matrix.indicator(cV)
    elif cv.ndim == 2:
        Z = cV;
    n_reg = Z.shape[1]
    
    # Get data 
    Y = Data.measurements
    N = Y.shape[0]
    YY = Y @ Y.T

    # Initialize fixed effects 
    X = None

    #  Depending on the way of dealing with the run effect, set up matrices and noise
    if run_effect == 'fixed':
        X = pcm_indicatorMatrix(Data.obs_descriptors['partition']);
    if run_effect == 'none' or run_effect == 'fixed':
        if noice_cov is None:
            Noise = BlockPlusIndepNoise(Data.obs_descriptors['partition'])
        else: 
            raise(NameError('not implemented'))
    if run_effect == 'random':
        X = pcm_indicatorMatrix(Data.obs_descriptors['partition']);
        if noice_cov is None:
            Noise = BlockPlusIndepNoise(Data.obs_descriptors['partition'])
        else: 
            raise(NameError('not implemented'))

    # Estimate noise parameters starting values
    Noise.get_theta0(Y,X,Z)
