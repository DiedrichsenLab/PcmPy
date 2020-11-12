"""
Regression module contains bare-bones version of  the PCM toolbox that can be used to tune ridge/Tikhonov coefficients in the context of tranditional regression models. No assumption are made about independent data partitions.
"""

import numpy as np
from numpy.linalg import solve, eigh, cholesky
from numpy import sum, diag, log, eye, exp, trace, einsum
import PcmPy as pcm

def likelihood_diagYYT(theta, Z, YY, num_var, comp, X=None, Noise=pcm.model.IndependentNoise(), return_deriv=0):
    """
    Negative Log-Likelihood of the data and derivative in respect to the parameters. This function is faster when P>N

    Parameters:
        theta (np.array)
            Vector of (log-)model parameters: These include model, signal and noise parameters
        Z (2d-np.array)
            Design matrix for random effects NxQ
        YY  (2d-np.array)
            NxN Matrix: Outer product of the data
        num_var (int)
            Number of variables in data set (columns of Y)
        comp (1d-np.array or list)
            Q-length: Indicates for each column of Z, which theta will be used for the weighting
        X (np.array)
            Fixed effects design matrix - will be accounted for by ReML
        Noise (pcm.Noisemodel)
            Pcm-noise mode to model block-effects (default: Indentity)
        return_deriv (int)
            0: Only return negative loglikelihood
            1: Return first derivative
            2: Return first and second derivative (default)

    Returns:
        negloglike:
            Negative log-likelihood of the data under a model
        dLdtheta (1d-np.array)
            First derivative of negloglike in respect to the parameters
        ddLdtheta2 (2d-np.array)
            Second derivative of negloglike in respect to the parameters

    """
    N = YY.shape[0]
    num_comp = max(comp)+1
    n_param = theta.shape[0]

    # Sort the parameters in model and noise paramaters
    model_params = theta[range(num_comp)]
    noise_params = theta[num_comp:]


    # Matrix inversion lemma. The following statement is the same as
    # V   = (Z*Z'*exp(theta) + S(noiseParam));
    # iV  = pinv(V);

    iS = Noise.inverse(noise_params)
    iG = 1 / exp(model_params[comp]) # Diagonal of the Inverse of G
    if type(iS) is np.float64:
        matrixInv = np.diag(iG) + Z.T @ Z * iS # Inner Matrix
        iV = (eye(N) - Z @ solve(matrixInv, Z.T) * iS) * iS
    else:
        matrixInv = np.diag(iG) + Z.T @ iS @ Z
        iV = iS - iS @ Z @ solve(matrixInv,Z.T) @ iS
    # For ReML, compute the modified inverse iVr
    if X is not None:
        iVX   = iV @ X
        iVr   = iV - iVX @ solve(X.T @ iVX, iVX.T)
    else:
        iVr = iV

    # Computation of (restricted) likelihood
    B = YY @ iVr
    ldet = -2 * sum(log(diag(cholesky(iV)))) # Safe computation
    llik = -num_var / 2 * ldet - 0.5 * einsum('ii->',B) # trace B
    if X is not None:
        # P/2 log(det(X'V^-1*X))
        llik -= num_var * sum(log(diag(cholesky(X.T @ iV @X)))) #

    # If no derivative - exit here
    if return_deriv == 0:
        return (-llik,) # Return as tuple for consistency

    # Calculate the first derivative
    iVdV = []

    # Get the quantity iVdV = inv(V)dVdtheta for model parameters
    for i,theta in enumerate(model_params):
        iVdV.append(iVr @ Z[:,comp==i] @ Z[:,comp==i].T * exp(theta))

    # Get iVdV for Noise parameters
    for j,theta in enumerate(noise_params):
        dVdtheta = Noise.derivative(noise_params,j)
        if type(dVdtheta) is np.float64:
            iVdV.append(iVr * dVdtheta)
        else:
            iVdV.append(iVr @ dVdtheta)

    # Based on iVdV we can get he first derivative
    dLdtheta = np.zeros((n_param,))
    for i in range(n_param):
        dLdtheta[i] = -num_var / 2 * trace(iVdV[i]) + 0.5 * einsum('ij,ij->',iVdV[i], B)

    # If only first derivative, exit here
    if return_deriv == 1:
        return (-llik, -dLdtheta)

    # Calculate expected second derivative
    d2L = np.zeros((n_param,n_param))
    for i in range(n_param):
        for j in range(i, n_param):
            d2L[i, j] = -num_var / 2 * einsum('ij,ij->',iVdV[i],iVdV[j])
            d2L[j, i] = d2L[i, j]

    if return_deriv == 2:
        return (-llik, -dLdtheta, -d2L)
    else:
        raise NameError('return_deriv needs to be 0, 1 or 2')

def likelihood_diagYTY(theta, Z, Y, comp, X=None, Noise=pcm.model.IndependentNoise(), return_deriv=0):
    """
    Negative Log-Likelihood of the data and derivative in respect to the parameters. This function is faster when N>>P.

    Parameters:
        theta (np.array)
            Vector of (log-)model parameters: These include model, signal and noise parameters
        Z (2d-np.array)
            Design matrix for random effects NxQ
        Y  (2d-np.array)
            NxP Matrix of data
        comp (1d-np.array or list)
            Q-length: Indicates for each column of Z, which theta will be used for the weighting
        X (np.array)
            Fixed effects design matrix - will be accounted for by ReML
        Noise (pcm.Noisemodel)
            Pcm-noise mode to model block-effects (default: Indentity)
        return_deriv (int)
            0: Only return negative loglikelihood
            1: Return first derivative
            2: Return first and second derivative (default)

    Returns:
        negloglike:
            Negative log-likelihood of the data under a model
        dLdtheta (1d-np.array)
            First derivative of negloglike in respect to the parameters
        ddLdtheta2 (2d-np.array)
            Second derivative of negloglike in respect to the parameters

    """
    N, num_var = Y.shape
    num_comp = max(comp)+1
    n_param = theta.shape[0]

    # Sort the parameters in model and noise paramaters
    model_params = theta[range(num_comp)]
    noise_params = theta[num_comp:]


    # Matrix inversion lemma. The following statement is the same as
    # V   = (Z*Z'*exp(theta) + S(noiseParam));
    # iV  = pinv(V);

    iS = Noise.inverse(noise_params)
    iG = 1 / exp(model_params[comp]) # Diagonal of the Inverse of G
    if type(iS) is np.float64:
        matrixInv = np.diag(iG) + Z.T @ Z * iS # Inner Matrix
        iV = (eye(N) - Z @ solve(matrixInv, Z.T) * iS) * iS
    else:
        matrixInv = np.diag(iG) + Z.T @ iS @ Z
        iV = iS - iS @ Z @ solve(matrixInv,Z.T) @ iS
    # For ReML, compute the modified inverse iVr
    if X is not None:
        iVX   = iV @ X
        iVr   = iV - iVX @ solve(X.T @ iVX, iVX.T)
    else:
        iVr = iV

    # Computation of (restricted) likelihood
    YiVr = Y.T @ iVr
    ldet = -2 * sum(log(diag(cholesky(iV)))) # Safe computation
    llik = -num_var / 2 * ldet - 0.5 * np.einsum('ij,ji',YiVr,Y) # trace(Y.T iVr Y)
    if X is not None:
        # P/2 log(det(X'V^-1*X))
        llik -= num_var * sum(log(diag(cholesky(X.T @ iV @X)))) #

    # If no derivative - exit here
    if return_deriv == 0:
        return (-llik,) # Return as tuple for consistency

    # Calculate the first derivative
    iVdV = []

    # Get the quantity iVdV = inv(V)dVdtheta for model parameters
    for i,theta in enumerate(model_params):
        iVdV.append(iVr @ Z[:,comp==i] @ Z[:,comp==i].T * exp(theta))

    # Get iVdV for Noise parameters
    for j,theta in enumerate(noise_params):
        dVdtheta = Noise.derivative(noise_params,j)
        if type(dVdtheta) is np.float64:
            iVdV.append(iVr * dVdtheta)
        else:
            iVdV.append(iVr @ dVdtheta)

    # Based on iVdV we can get he first derivative
    # Last term is 
    #     0.5 trace(Y.T @ iVr @ dV @ iVr @ Y)
    #  =  0.5 trace(Y.T @ iVdV @ iVr @ Y)
    dLdtheta = np.zeros((n_param,))
    for i in range(n_param):
        dLdtheta[i] = -num_var / 2 * trace(iVdV[i]) + 0.5 * einsum('ij,ij->',Y.T @ iVdV[i], YiVr)

    # If only first derivative, exit here
    if return_deriv == 1:
        return (-llik, -dLdtheta)

    # Calculate expected second derivative
    d2L = np.zeros((n_param,n_param))
    for i in range(n_param):
        for j in range(i, n_param):
            d2L[i, j] = -num_var / 2 * einsum('ij,ij->',iVdV[i],iVdV[j])
            d2L[j, i] = d2L[i, j]

    if return_deriv == 2:
        return (-llik, -dLdtheta, -d2L)
    else:
        raise NameError('return_deriv needs to be 0, 1 or 2')

class RidgeDiag:
    """
        Class for Linear Regression with Tikhonov (L2) regularization.
        The regularization matrix for this class is diagnonal, with groups
        of elements along the diagonal sharing the same Regularisation factor.
    """
    def __init__(self, components, theta0 = None, fit_intercept =  True, noise_model = pcm.model.IndependentNoise(),like_fcn = None):
        self.components = components
        self.noise_model = noise_model
        self.n_param = max(components)+1+self.noise_model.n_param
        self.noise_idx = np.arange(max(components)+1,self.n_param)
        self.theta0_ = np.zeros((self.n_param,)) # Empty theta0
        self.theta_  = np.zeros((self.n_param,))
        self.fit_intercept = fit_intercept

    def optimize_regularization(self, Z , Y, X = None, optim_param = {}, like_fcn = 'auto'):
        """
        Optimizar the
        Parameters:
            Z (2d-np.array)
                Design matrix for random effects NxQ
            Y  (2d-np.array)
                NxP Matrix of data
            comp (1d-np.array or list)
                Q-length: Indicates for each column of Z, which theta will be used for the weighting
            X (np.array)
                Fixed effects design matrix - will be accounted for by ReML
            Noise (pcm.Noisemodel)
                Pcm-noise mode to model block-effects (default: Indentity)
        Returns:
            theta  (1d-np.array)
        """
        Z, X = self.add_intercept(Z, X)
        N, P = Y.shape
        if like_fcn == 'auto':
            if N > 7 * P:  # This is likely a crude approximation
                like_fcn = 'YTY'
            else:
                like_fcn = 'YYT'
        
        if like_fcn == 'YTY':
            fcn = lambda x: likelihood_diagYTY(x, Z, Y, self.components, X, self.noise_model, return_deriv=2)
        elif like_fcn == 'YYT':
            YY = Y @ Y.T
            fcn = lambda x: likelihood_diagYYT(x, Z, YY, P, self.components, X, self.noise_model, return_deriv=2)
        else:
            raise NameError('like_fcn needs to be auto, YYT, or YTY')
        self.theta_, self.trainLogLike_, self.optim_info = pcm.optimize.newton(self.theta0_, fcn, **optim_param)
        return self

    def fit(self, Z ,Y , X = None):
        N = Z.shape[0]
        Z, X = self.add_intercept(Z, X)
        # Get the inverse of the covariance matrix
        G = exp(self.theta_[self.components]) # Diagonal of the Inverse of G
        iS = self.noise_model.inverse(self.theta_[self.noise_idx])
        if type(iS) is np.float64:
            matrixInv = np.diag(1/G) + Z.T @ Z * iS # Inner Matrix
            iV = (eye(N) - Z @ solve(matrixInv, Z.T) * iS) * iS
        else:
            matrixInv = np.diag(1/G) + Z.T @ iS @ Z
            iV = iS - iS @ Z @ solve(matrixInv,Z.T) @ iS

        # If any fixed effects are given, estimate them and modify residual forming matrix
        if X is not None:
            iVX   = iV @ X
            P     = solve(X.T @ iVX, iVX.T)
            self.beta_ = P @ Y
            Yr = Y - X @ self.beta_
        else:
            Yr = Y

        # Estimate the random effects over solve
        self.coef_ = np.diag(G) @ Z.T @ iV @ Yr
        return self

    def predict(self, Z, X = None):
        self.add_intercept(Z, X)
        if (X is None):
            Yp = Z @ self.coef_
        else:
            Yp = Z @ self.coef_ + X @ self.beta_
        return Yp

    def set_params(self,params):
        pass

    def get_params(self,params):
        pass

    def add_intercept(self, Z, X = None):
        N  = Z.shape[0]
        if (self.fit_intercept):
            if (X is None):
                X = np.ones((N,1))
            else:
                X = np.c_[X - np.mean(X,axis=0),np.ones((N,))]
            Z = Z - np.mean(Z,axis=0)
        return Z, X


