"""
Optimization module for PCM toolbox with main functionality for model fitting. 
    newton
        Newton-Raphson like method with automatic regularization of the second derivative
@author: jdiedrichsen
"""

import numpy as np
from numpy.linalg import solve, eigh, cholesky
from numpy import sum, diag, log, eye, exp, trace, einsum
from PcmPy import model

def newton(theta0, lossfcn, max_iter=80, thres= 1e-4, hess_reg=0.01, 
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

def best_algorithm(M, algorithm=None):
    """

    Parameters:
        M (List of pcm.Model)
        algorithm (string)
            Overwrite for algorithm 
    """
    pass