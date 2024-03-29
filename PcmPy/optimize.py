"""
Optimization module for PCM toolbox with main functionality for model fitting.
@author: jdiedrichsen
"""

import numpy as np
from numpy.linalg import solve, eigh, cholesky
from numpy import sum, diag, log, eye, exp, trace, einsum
from PcmPy import model

def newton(theta0, lossfcn, max_iter=80, thres= 1e-4, hess_reg=1e-4,
             regularization='sEig',verbose=0, fit_param = None):
    """
    Minimize a loss function using Newton-Raphson with automatic regularization

    Parameters:
        theta (np.array)
            Vector of parameter starting values
        lossfcn (fcn)
            Handle to loss function that needs to return
            a) Loss (Negative log-likelihood)
            b) First derivative of the Loss
            c) Expected second derivative of the loss
        max_iter (int)
            Maximal number of iterations (default: 80)
        thres (float)
            Threshold for change in Loss function (default: 1e-4)
        hess_reg (float)
            starting regulariser on the Hessian matrix (default 1e-4)
        regularization (string)
            'L': Levenberg
            'LM': Levenberg-Marquardt
            'sEig':smallest Eigenvalue (default)
        verbose (int)
             0: No feedback,
             1:Important warnings
             2:full feedback regularisation
        fit_param (Logical)
            If provided, it will only fit the parameters indicated
    Returns:
            theta (np.array)
                theta at minimum
            loss (float)
                minimal loss
            info (dict)
                Dictionary with more information abuot the fit
    """
    if fit_param is None:
        fit_param = np.ones(theta0.shape,dtype=bool)
    # Initialize Interations
    dF = np.Inf
    H = theta0.shape[0] # Number of parameters
    theta = theta0
    dL = np.Inf
    thetaH = np.empty((theta.shape[0], max_iter))
    thetaH[:] = np.nan
    regH = np.empty((max_iter,)) * np.nan
    nl = np.empty((max_iter,)) * np.nan
    for k in range(max_iter):
        # If more than the first interation: Try to update theta
        if k > 0:
            # Fisher scoring: update dh = inv(ddF/dhh)*dF/dh
            # if it fails increase regularisation until dFdhh is invertible
            # Add regularisation to second derivative
            # ----------------------------------------------------------------------
            while True:
                try:
                    if (regularization == 'LM'): # Levenberg-Marquardt
                        H = dFdhh + diag(diag(dFdhh)) * hess_reg
                        dtheta = solve(H,dFdh)
                    elif (regularization == 'L'): # Levenberg
                        H = dFdhh + eye(dFdhh.shape[0]) * hess_reg
                        dtheta = solve(H,dFdh)
                    elif (regularization == 'sEig'):
                        [lH,VH]=eigh(dFdhh)
                        lH[lH<hess_reg] = hess_reg  # Increase the smallest
                        dtheta = (VH * 1/lH) @ VH.T @ dFdh
                    break
                except ValueError: # Any error
                    if verbose == 2:
                        print('Ill-conditioned Hessian. Regularisation %2.3f\n',hess_reg)
                    hess_reg = hess_reg*10
                    if hess_reg > 100000:
                        if verbose > 0:
                            print('Cant regularise second derivative.. Giving up\n')
                        exitflag=3 # Regularisation increased too much
                        break # Give up
            theta[fit_param] = theta[fit_param] - dtheta[fit_param]

        # Record the current theta
        thetaH[:,k] = theta
        regH[k] = hess_reg

        # Evaluate the current likelihood
        try:
            nl[k], dFdh, dFdhh = lossfcn(theta)
        except np.linalg.LinAlgError:
            if (k==0):
                raise(NameError('Bad starting values - failed likelihood'))
            else:
                nl[k]=np.Inf
        # Safety check if negative likelihood decreased
        if ((k>0 and (nl[k] - nl[k-1])>1e-16)) or np.isnan(nl[k]) or np.any(np.isnan(dFdh)) or np.any(np.isnan(dFdhh)):
            hess_reg = hess_reg * 10 # Increase regularisation
            if verbose == 2:
                print('Step back. Regularisation:',hess_reg)
            theta = thetaH[:,k-1]
            thetaH[:,k] = theta
            nl[k] = nl[k-1]
            dFdh = dFdh_old
            dFdhh = dFdhh_old
            dL = np.Inf # Definitely try again
        else:
            if hess_reg > 1e-8:
                hess_reg = hess_reg / 10 # Decrease regularization
            if k > 0:
                dL = nl[k-1] - nl[k]
        # Record current first and second derivative if we need to take a step back
        dFdh_old = dFdh
        dFdhh_old = dFdhh

        if dL < thres:
            break
    # Make an info dict
    INFO = {'iter': k, 'reg': hess_reg,
            'thetaH': thetaH[:,:k+1], 'regH': regH[:k+1],'loglikH': -nl[:k+1]}
    # Return likelihood
    return theta, -nl[k], INFO

def best_algorithm(M, algorithm=None):
    """

    Parameters:
        M (List of pcm.Model)
        algorithm (string)
            Overwrite for algorithm
    """
    pass

def mcmc(th0,likelihood_fcn,
         proposal_sd=0.1,
         burn_in=100,
         n_samples=1000,
         verbose=1):
    """Implement Markov Chain Monte Carlo sampling for PCM models
    Metropolis-Hastings algorithm with adaptive proposal distribution
    """
    n_params = th0.shape[0]
    TH = np.empty((n_params,n_samples),dtype=float)
    l = np.zeros((n_samples,),dtype=float)
    # Set the starting values:
    TH[:,0]= th0
    l[0] = -likelihood_fcn(th0)[0]
    rejections = 0
    # Generate proposal e-masse:
    dth = np.random.normal(np.zeros((1,n_samples)),proposal_sd.reshape(-1,1))
    r = np.random.rand(n_samples)

    # Run the chain
    for i in range(n_samples-1):
        if verbose > 0:
            if np.mod(i+1,1000) == 0:
                print(f'{i+1} samples {rejections/i*100:.1f}% rejection')
        th_p = TH[:,i]+dth[:,i]
        l_p = -likelihood_fcn(th_p)[0]
        if l_p > l[i]:
            TH[:,i+1] = th_p
            l[i+1] = l_p
        else:
            if r[i] < np.exp(l_p-l[i]):
                TH[:,i+1] = th_p
                l[i+1] = l_p
            else:
                TH[:,i+1] = TH[:,i]
                l[i+1] = l[i]
                rejections += 1
    return TH,l

