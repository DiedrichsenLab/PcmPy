"""
Inference module for PCM toolbox with main functionality for model fitting and evaluation.
    likelihood_individ: Likelihood for an individual data set
    likelihood_group: Likelihood with shared model parameters across group

@author: jdiedrichsen
"""

import numpy as np
from numpy.linalg import solve, eigh, cholesky
from numpy import sum, diag, log, eye
from PcmPy import model


def likelihood_individ(theta, M, YY, Z, X=None, 
                       Noise=model.IndependentNoise(),
                       num_channels=1, fit_scale=False, return_deriv=0):
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
        num_channels (int)
            Number of channels
        fit_scale (bool)
            Fit a scaling parameter for the model (default is False)
        return_deriv (int)
            0: Do not return any derivative
            1: Return first derivative
            2: Return first and second derivative (default)

    """
    N,NC = YY.shape
    Q = Z.shape[1]

    # Get G-matrix and derivative of G-matrix in respect to parameters
    model_params = theta[range(M.n_param)]
    G,dGdtheta = M.predict(model_params)

    # Get the scale parameter and scale G by it
    if fit_scale:
        scale_param = theta[model_params+1]
    else:
        scale_param = 1
    Gs = G * np.exp(scale_param)

    # Get the noise model parameters and noise prediction
    noise_params = theta[M.n_param+fit_scale:]
    S = Noise.predict(noise_params)

    Gs = (Gs + Gs.T) / 2 # Symmetrize
    s, U = eigh(Gs)
    idx = s > (10e-10) # Increased to 10*eps from 2*eps
    Zu = Z @ U[:, idx]

    # Apply the matrix inversion lemma. The following statement is the same as
    # V   = (Z*Gs*Z' + S(noiseParam));
    # iV  = pinv(V);

    iS = Noise.inverse(noise_params)
    if type(iS) is float:
        iV = (eye(N) - Zu / (diag(1./dS(idx)) * iS +Zu.T @ Zu) @ Zu.T) * iS
    else:
        iV = iS - iS @ Zu / (diag(1./dS(idx)) +Zu.T @ iS @ Zu) @ Zu.T @ iS
    # For ReML, compute the modified inverse iVr
    if X is not None:
        iVX   = iV @ X
        iVr   = iV - iVX @ solve(X.T @ iVX, iVX.T)
    else: 
        iVr = iV

    # Computation of (restricted) likelihood
    ldet = -2 * sum(log(diag(linalg.chol(iV)))) # Safe computation 
    llik = -P/2 * ldet - 0.5 * sum(iVr * YY)
    if X is not None:
        # P/2 log(det(X'V^-1*X))
        llik = llik - P * sum(log(diag(cholesky(X.T @ iV @X)))) #
    negLogLike = -llik  # Invert sign

    if return_deriv == 0: 
        return negLogLike

    # Calculate the first derivative
    # A = iVr @  Z
    # B = YY @ iVr
    #  Get the derivatives for all the parameters
    # for i in range(Model.n_param):
            #iVdV{i} = A*pcm_blockdiag(dGdtheta(:,:,i),zeros(numRuns))*Z'*exp(scaleParam);
            #dLdtheta(i,1) = -P/2*trace(iVdV{i})+1/2*traceABtrans(iVdV{i},B);
        # Get the derivatives for the Noise parameters
        # i = M.numGparams+1;  % Which number parameter is it?
        # if (isempty(OPT.S))
        #     dVdtheta{i}          = eye(N)*exp(noiseParam);
        # else
        #     dVdtheta{i}          = OPT.S.S*exp(noiseParam);
        # end;
        # iVdV{i}     = iVr*dVdtheta{i};
        # dLdtheta(i,1) = -P/2*trace(iVdV{i})+1/2*traceABtrans(iVdV{i},B);
    """
     % Get the derivatives for the scaling parameters
     if (OPT.fitScale)
        i = M.numGparams+2;    % Which number parameter is it?
        iVdV{i}          = A*pcm_blockdiag(G,zeros(numRuns))*Z'*exp(scaleParam);
        dLdtheta(i,1)    = -P/2*trace(iVdV{i})+1/2*traceABtrans(iVdV{i},B);
        % dLdtheta(i,1)    = dLdtheta(i,s) - scaleParam/OPT.scalePrior; % add prior to scale parameter
     end;


    % Get the derivatives for the block parameter
    if (~isempty(OPT.runEffect) && ~isempty(OPT.runEffect))
        i = M.numGparams+2+OPT.fitScale;  % Which number parameter is it?
        %C          = A*pcm_blockdiag(zeros(size(G,1)),eye(numRuns));
        iVdV{i}     = A*pcm_blockdiag(zeros(K),eye(numRuns))*Z'*exp(runParam);
        dLdtheta(i,1) = -P/2*trace(iVdV{i})+1/2*traceABtrans(iVdV{i},B);
    end;

    % invert sign
    dnl   = -dLdtheta;
    numTheta=i;
    end;

    % Calculate expected second derivative?
    if (nargout>2)
    for i=1:numTheta
        for j=i:numTheta;
            d2nl(i,j)=-P/2*traceABtrans(iVdV{i},iVdV{j});
            d2nl(j,i)=d2nl(i,j);
        end;
    end;
    d2nl=-d2nl;
    end;
    """

