"""
Inference module for PCM toolbox with main functionality for model fitting and evaluation.
    likelihood_individ: Likelihood for an individual data set
    likelihood_group: Likelihood with shared model parameters across group

@author: jdiedrichsen
"""

def likelihood_individ(theta, Model, YY, Z, X=None, Noise=None, num_channels=1, fit_scale=False, return_deriv=2):
    """
    Negative Log-Likelihood of the data and derivative in respect to the parameters

    Parameters:
        theta (np.array)
            Vector of (log-)model parameters: These include model, signal and noise parameters
        Model (pcm.Model)
            Model object with predict function
        YY (2d-np.array)
            NxN Matrix of outer product of the activity data (Y*Y')
        Z (2d-np.array)
            NxQ Design matrix - relating the trials (N) to the random effects (Q)
        X (np.array)
            Fixed effects design matrix - will be accounted for by ReML
        Noise (pcm.Noisemodel)
            Optional pcm-noise mode to model block-effects
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
    model_params = theta[range(Model.n_param)]
    G,dGdtheta = Model.predict(model_params)

    # Get the scale parameter
    if fit_scale:
        scale_param = theta[model_params+1]
    else:
        scale_param = 1
    Gs = G*np.exp(scale_param);         % Scale the subject G matrix up by 

    # Get the noise model parameters and noise prediction
    noise_params = theta[M.n_param+fit_scale,:]

    # 
individual scale parameter

if (~isempty(OPT.runEffect))
    numRuns = size(OPT.runEffect,2);
    runParam = theta(M.numGparams+2+OPT.fitScale);    % Subject run effect parameter
    Gs = pcm_blockdiag(Gs,eye(numRuns)*exp(runParam));  % Include run effect in G
    Z = [Z OPT.runEffect];                 % Include run effect in design matrix
else
    numRuns = 0;                                % No run effects modelled
end;


% Find the inverse of V - while dropping the zero dimensions in G
Gs = (Gs+Gs')/2;        % Symmetrize
[u,s] = eig(Gs);
dS    = diag(s);
idx   = dS>(10*eps); % Increased to 10*eps from 2*eps
Zu     = Z*u(:,idx);

% Apply the matrix inversion lemma. The following statement is the same as
% V   = (Z*Gs*Z' + S.S*exp(noiseParam));
% iV  = pinv(V);
if (isempty(OPT.S))
    iV    = (eye(N)-Zu/(diag(1./dS(idx))*exp(noiseParam)+Zu'*Zu)*Zu')./exp(noiseParam); % Matrix inversion lemma
else
    iV    = (OPT.S.invS-OPT.S.invS*Zu/(diag(1./dS(idx))*exp(noiseParam)+Zu'*OPT.S.invS*Zu)*Zu'*OPT.S.invS)./exp(noiseParam); % Matrix inversion lemma
end
iV  = real(iV); % sometimes iV gets complex

% For ReML, compute the modified inverse iVr
if (~isempty(X))
    iVX   = iV * X;
    iVr   = iV - iVX*((X'*iVX)\iVX');
else
    iVr   = iV;
end

% Computation of (restricted) likelihood
ldet  = -2* sum(log(diag(chol(iV))));        % Safe computation of the log determinant (V) Thanks to code from D. lu
l     = -P/2*(ldet)-0.5*traceABtrans(iVr,YY);
if (~isempty(X)) % Correct for ReML estimates
    l = l - P*sum(log(diag(chol(X'*iV*X))));  % - P/2 log(det(X'V^-1*X));
end
negLogLike = -l; % Invert sign


% Calculate the first derivative
if (nargout>1)
    A     = iVr*Z;      % Precompute some matrices
    B     = YY*iVr;
    % Get the derivatives for all the parameters
    for i = 1:M.numGparams
        iVdV{i} = A*pcm_blockdiag(dGdtheta(:,:,i),zeros(numRuns))*Z'*exp(scaleParam);
        dLdtheta(i,1) = -P/2*trace(iVdV{i})+1/2*traceABtrans(iVdV{i},B);
    end

    % Get the derivatives for the Noise parameters
    i = M.numGparams+1;  % Which number parameter is it?
    if (isempty(OPT.S))
        dVdtheta{i}          = eye(N)*exp(noiseParam);
    else
        dVdtheta{i}          = OPT.S.S*exp(noiseParam);
    end;
    iVdV{i}     = iVr*dVdtheta{i};
    dLdtheta(i,1) = -P/2*trace(iVdV{i})+1/2*traceABtrans(iVdV{i},B);

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
