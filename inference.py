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

def optim_nr(theta0, lossfcn, max_iter=80, thres= 1e-4, hess_reg=0.01, 
             regularization='L',verbose=0):
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
            starting regulariser on the Hessian matrix (default 1)
        regularization (string)
            'L': Levenberg
            'LM': Levenberg-Marquardt
            'sEig':smallest Eigenvalue (default)
        verbose (int)
             0: No feedback,
             1:Important warnings
             2:full feedback regularisation
    Returns: 
            theta (np.array)
                theta at minimum 
            loss (float)
                minimal loss
            info (dict)
                Dictionary with more information abuot the fit
    """
    CATCHEXP = {'MATLAB:nearlySingularMatrix','MATLAB:singularMatrix',...
    'MATLAB:illConditionedMatrix','MATLAB:posdef',...
    'MATLAB:nearlySingularMatrix','MATLAB:eig:matrixWithNaNInf'};

    # Initialize Interations
    dF = np.Inf;
    H = theta0.shape[0]) # Number of parameters
    hess_reg = hess_reg * eye(H,H) # Regularization on Hessian 
    theta = theta0
    dL = inf
    for k in range(max_iter)
        # If more than the first interation: Try to update theta
        if k > 0:
            # Fisher scoring: update dh = inv(ddF/dhh)*dF/dh
            # if it fails increase regularisation until dFdhh is invertible
            # Add regularisation to second derivative
            # ----------------------------------------------------------------------
            while true
                try:
                    if (regularization == 'LM'): # Levenberg-Marquardt
                        H = dFdhh + diag(diag(dFdhh)) * hess_reg
                        dtheta = solve(H,dFdh)
                    elif (regularization == 'L'): # Levenberg
                        H = dFdhh + eye(dFdhh.shape[0]) * hess_reg
                        dtheta = solve(H,dFdh)
                    elif (regularizaiton == 'sEig'): 
                        [LH,VH]=eigh(dFdhh)
                        lH(lH<hess_reg) = hess_reg  # Increase the smallest 
                        dtheta = (VH * 1/lh) @ VH.T @ dFdh; 
                except ValueError: # Any error  
                        if verbose == 2:
                            print('Ill-conditioned Hessian. Regularisation %2.3f\n',hess_reg)
                        hess_reg = hess_reg*10
                        if hess_reg > 100000:
                            if verbose > 0:
                                print('Cant regularise second derivative.. Giving up\n')
                            exitflag=3 # Regularisation increased too much 
                            break # Give up
            theta = theta - dtheta

        # Record the current theta 
        thetaH[:,k] = theta
        regH[k] = hess_reg

        # Evaluate the current likelihood 
        nl[k], dFdh, dFdhh = likefcn(theta)
        #except: # Catch errors based on invalid parameter settings
        #if any(strcmp(ME.identifier,CATCHEXP))
        #    if (k==1)
        #        error('bad initial values for theta');
        #    else
        #        nl(k)=inf;         % Set new likelihood to -inf: take a step #back
        #    end;
        #else
        #    ME.rethrow;
        #end;
        # Safety check if negative likelihood decreased
        if (k>1 and (nl[k] - nl[k-1])>1e-16):
            hess_reg = hess_reg * 10 # Increase regularisation
            if verbose == 2:
                printf('Step back. Regularisation %2.3f\n',hess_reg)
            theta = thetaH(:,k-1);
            thetaH[:,k]=theta;
            nl[k]=nl[k-1]
            dFdh = dFdh_old
            dFdhh = dFdhh_old
            dL = inf # Definitely try again
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
    
    # Return likelihood
    return theta, -likefcn(theta)
