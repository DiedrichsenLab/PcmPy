"""
Regression module contains bare-bones version of  the PCM toolbox that can be used to tune ridge/Tikhonov coefficients in the context of tranditional regression models. No assumption are made about independent data partitions.
"""

import numpy as np
from numpy.linalg import solve, eigh, cholesky
from numpy import sum, diag, log, eye, exp, trace, einsum, sqrt
import PcmPy as pcm

def compute_iVr(Z, G, iS, X = None):
    """Fast inverse of V matrix using the matrix inversion lemma

    Parameters:
        Z (2d-np.array)
            Design matrix for random effects NxQ
        G  (1d or 2d-np.array)
            Q x Q Matrix: variance of random effect
        iS (scalar or NxN matrix)
            Inverse variance of noise matrix
        X (2d-np.array)
            Design matrix for random effects

    Returns:
        iV  (2d-np.array)
            inv(Z*G*Z' + S);
        iVr (2d-np.array)
            iV - iV * X inv(X' * iV *X) * X' *iV
        ldet (scalar)
            log(det(iV))

    """

    N = Z.shape[0]
    idx = G > (10e-10) # Find sufficiently large weights
    iG = 1 / G[idx]
    Zr = Z[:,idx]
    if type(iS) is np.float64: # For i.i.d noise use fast solution
        matrixInv = np.diag(iG) + Zr.T @ Zr * iS # Inner Matrix
        iV = (eye(N) - Zr @ solve(matrixInv, Zr.T) * iS) * iS
        Zw = Zr * sqrt(G[idx]) # Weighted Z
        lam,_ = eigh(Zw.T @ Zw)
        ldet = sum(log(lam+1/iS)) - (N-sum(idx))*log(iS) # Shortcut to log-determinant
    else:                       # For non-i.i.d noise use slower solution
        matrixInv = np.diag(iG) + Zr.T @ iS @ Zr
        iV = iS - iS @ Zr @ solve(matrixInv,Zr.T) @ iS
        ldet = -2 * sum(log(diag(cholesky(iV)))) # Safe computation
    # For ReML, compute the modified inverse iVr
    if X is not None:
        iVX   = iV @ X
        iVr   = iV - iVX @ solve(X.T @ iVX, iVX.T)
    else:
        iVr = iV
    return (iV, iVr, ldet)

def likelihood_diagYYT_ZZT(theta, Z, YY, num_var, comp, X=None, Noise=pcm.model.IndependentNoise(), return_deriv=0):
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

    # Calculate iV and iVr via Matrix inversion lemma.
    iS = Noise.inverse(noise_params)
    G  = exp(model_params[comp])
    iV, iVr, ldet = compute_iVr(Z, G, iS, X)

    # Computation of (restricted) likelihood
    B = YY @ iVr
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
        dLdtheta[i] = -num_var / 2 * trace(iVdV[i]) + 0.5 * einsum('ij,ij->',iVdV[i], B) # Trace(A@B.T)

    # If only first derivative, exit here
    if return_deriv == 1:
        return (-llik, -dLdtheta)

    # Calculate expected second derivative
    d2L = np.zeros((n_param,n_param))
    for i in range(n_param):
        for j in range(i, n_param):
            d2L[i, j] = -num_var / 2 * einsum('ij,ji->',iVdV[i],iVdV[j]) # Trace(A@B)
            d2L[j, i] = d2L[i, j]

    if return_deriv == 2:
        return (-llik, -dLdtheta, -d2L)
    else:
        raise NameError('return_deriv needs to be 0, 1 or 2')

def likelihood_diagYYT_ZTZ(theta, Z, YY, num_var, comp, X=None, Noise=pcm.model.IndependentNoise(), return_deriv=0):
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

    # Calculate iV and iVr via Matrix inversion lemma.
    iS = Noise.inverse(noise_params)
    G  = exp(model_params[comp])
    iV, iVr, ldet = compute_iVr(Z, G, iS, X)

    # Computation of (restricted) likelihood
    B = YY @ iVr
    llik = -num_var / 2 * ldet - 0.5 * einsum('ii->',B) # trace B
    if X is not None:
        # P/2 log(det(X'V^-1*X))
        llik -= num_var * sum(log(diag(cholesky(X.T @ iV @X)))) #

    # If no derivative - exit here
    if return_deriv == 0:
        return (-llik,) # Return as tuple for consistency

    # Calculate the first derivative
    iVZ  = []
    # Get the quantity iVdV = inv(V)dVdtheta for model parameters
    for i,th in enumerate(model_params):
        iVZ.append(iVr @ Z[:,comp==i])
    # Get iVdV for Noise parameters
    for j,th in enumerate(noise_params):
        dVdtheta = Noise.derivative(noise_params,j)
        if type(dVdtheta) is np.float64:
            iVZ.append(iVr * dVdtheta)
        else:
            iVZ.append(iVr @ dVdtheta)

    # Based on iVdV we can get he first derivative
    # Last term is
    #     0.5 trace(Y.T @ iVr @ dV @ iVr @ Y)
    #  =  0.5 trace(iVZ @ iVZ' @ Y @ Y.T )
    dLdtheta = np.zeros((n_param,))

    for i,th in enumerate(theta):
        if i < model_params.shape[0]:
            dLdtheta[i]= -exp(th)/2 * (num_var * einsum('ij,ij->',iVZ[i],Z[:,comp==i])-einsum('ij,ij->', iVZ[i] @ Z[:,comp==i].T, B))
        else:
            dLdtheta[i]= -num_var / 2 * trace(iVZ[i]) + 0.5 * einsum('ij,ij->', iVZ[i], B)

    # If only first derivative, exit here
    if return_deriv == 1:
        return (-llik, -dLdtheta)

    # Calculate expected second derivative
    d2L = np.zeros((n_param,n_param))
    for i in range(n_param):
        for j in range(i, n_param):
            if (j < model_params.shape[0]):
                d2L[i, j] = -num_var / 2 * einsum('ij,ji->',Z[:,comp==j].T @ iVZ[i], Z[:,comp==i].T @ iVZ[j]) * exp(theta[i]) * exp(theta[j])
            elif (i < model_params.shape[0]):
                d2L[i, j] = -num_var / 2 * einsum('ij,ji->',iVZ[i], Z[:,comp==i].T @ iVZ[j])* exp(theta[i])
            else:
                d2L[i, j] = -num_var / 2 * einsum('ij,ji->',iVZ[i], iVZ[j])
            d2L[j, i] = d2L[i, j]

    if return_deriv == 2:
        return (-llik, -dLdtheta, -d2L)
    else:
        raise NameError('return_deriv needs to be 0, 1 or 2')

def likelihood_diagYTY_ZZT(theta, Z, Y, comp, X=None, Noise=pcm.model.IndependentNoise(), return_deriv=0):
    """Negative Log-Likelihood of the data and derivative in respect to the parameters. This function is faster when N>>P.

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
    Q = Z.shape[1]
    num_comp = max(comp)+1
    n_param = theta.shape[0]

    # Sort the parameters in model and noise paramaters
    model_params = theta[range(num_comp)]
    noise_params = theta[num_comp:]

    # Calculate iV and iVr via Matrix inversion lemma.
    iS = Noise.inverse(noise_params)
    G  = exp(model_params[comp])
    iV, iVr, ldet = compute_iVr(Z, G, iS, X)

    # Computation of (restricted) likelihood
    YiVr = Y.T @ iVr
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
            d2L[i, j] = -num_var / 2 * einsum('ij,ji->',iVdV[i],iVdV[j])
            d2L[j, i] = d2L[i, j]

    if return_deriv == 2:
        return (-llik, -dLdtheta, -d2L)
    else:
        raise NameError('return_deriv needs to be 0, 1 or 2')

def likelihood_diagYTY_ZTZ(theta, Z, Y, comp, X=None, Noise=pcm.model.IndependentNoise(), return_deriv=0):
    N, num_var = Y.shape
    Q = Z.shape[1]
    num_comp = max(comp)+1
    n_param = theta.shape[0]

    # Sort the parameters in model and noise paramaters
    model_params = theta[range(num_comp)]
    noise_params = theta[num_comp:]

    # Calculate iV and iVr via Matrix inversion lemma.
    iS = Noise.inverse(noise_params)
    G  = exp(model_params[comp])
    iV, iVr, ldet = compute_iVr(Z, G, iS, X)

    # Computation of (restricted) likelihood
    YiVr = Y.T @ iVr
    llik = -num_var / 2 * ldet - 0.5 * np.einsum('ij,ji',YiVr,Y) # trace(Y.T iVr Y)
    if X is not None:         # P/2 log(det(X'V^-1*X))
        llik -= num_var * sum(log(diag(cholesky(X.T @ iV @X)))) #

    # If no derivative - exit here
    if return_deriv == 0:
        return (-llik,) # Return as tuple for consistency

    # Calculate the first derivative
    iVZ  = []
    # Get the quantity iVdV = inv(V)dVdtheta for model parameters
    for i,th in enumerate(model_params):
        iVZ.append(iVr @ Z[:,comp==i])
    # Get iVdV for Noise parameters
    for j,th in enumerate(noise_params):
        dVdtheta = Noise.derivative(noise_params,j)
        if type(dVdtheta) is np.float64:
            iVZ.append(iVr * dVdtheta)
        else:
            iVZ.append(iVr @ dVdtheta)

    # Based on iVdV we can get he first derivative
    # Last term is
    #     0.5 trace(Y.T @ iVr @ dV @ iVr @ Y)
    #  =  0.5 trace(Y.T @ iVdV @ iVr @ Y)
    dLdtheta = np.zeros((n_param,))

    for i,th in enumerate(theta):
        if i < model_params.shape[0]:
            A = Y.T @ iVZ[i]
            dLdtheta[i]= -exp(th)/2 * (num_var * einsum('ij,ij->',iVZ[i],Z[:,comp==i])-einsum('ij,ij->', A, A))
        else:
            dLdtheta[i]= -num_var / 2 * trace(iVZ[i]) + 0.5 * einsum('ij,ij->', Y.T @ iVZ[i], YiVr)

    # If only first derivative, exit here
    if return_deriv == 1:
        return (-llik, -dLdtheta)

    # Calculate expected second derivative
    d2L = np.zeros((n_param,n_param))
    for i in range(n_param):
        for j in range(i, n_param):
            if (j < model_params.shape[0]):
                d2L[i, j] = -num_var / 2 * einsum('ij,ji->',Z[:,comp==j].T @ iVZ[i], Z[:,comp==i].T @ iVZ[j]) * exp(theta[i]) * exp(theta[j])
            elif (i < model_params.shape[0]):
                d2L[i, j] = -num_var / 2 * einsum('ij,ji->',iVZ[i], Z[:,comp==i].T @ iVZ[j])* exp(theta[i])
            else:
                d2L[i, j] = -num_var / 2 * einsum('ij,ji->',iVZ[i], iVZ[j])
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

    def __init__(self, components, theta0 = None, fit_intercept =  True, noise_model = pcm.model.IndependentNoise()):
        """
        Constructor
            Parameters:
                components (1d-array like)
                    Indicator to which column of design matrix belongs to which group
                theta0 (1d  np.array)
                    Vector of of starting values for optimization
                fit_intercept (Boolean)
                    Should intercept be added to fixed effects (Dafault: true)
                noise_model (pcm.model.NoiseModel)
                    Model specifying the full-rank noise effects

        """
        self.components = components
        self.noise_model = noise_model
        self.n_param = max(components)+1+self.noise_model.n_param
        self.noise_idx = np.arange(max(components)+1,self.n_param)
        self.theta0_ = np.zeros((self.n_param,)) # Empty theta0
        self.theta_  = None
        self.fit_intercept = fit_intercept

    def optimize_regularization(self, Z , Y, X = None, optim_param = {}, like_fcn = 'auto'):
        """
        Optimizes the hyper parameters (regularisation) of the regression mode
        Parameters:
            Z (2d-np.array)
                Design matrix for random effects NxQ
            Y  (2d-np.array)
                NxP Matrix of data
            X (np.array)
                Fixed effects design matrix - will be accounted for by ReML
            optim_parameters (dictionary of parameters)
                parameters for the optimization routine
        Returns:
            self
                Model with fitted parameters

        """
        Z, X = self.add_intercept(Z, X)
        N, P = Y.shape
        N1, Q = Z.shape
        num_comp = max(self.components)+1

        if (N != N1):
            raise NameError('Y and Z need to have some shape[0]')
        if like_fcn == 'auto':
            if N < 1.5 * P:
                like_fcn = 'YYT'
            else:
                like_fcn = 'YTY'
            if Q/num_comp < N/2:
                like_fcn += '_ZTZ'
            else:
                like_fcn += '_ZZT'

        if like_fcn == 'YYT_ZZT':
            YY = Y @ Y.T
            fcn = lambda x: likelihood_diagYYT_ZZT(x, Z, YY, P, self.components, X, self.noise_model, return_deriv=2)
        elif like_fcn == 'YYT_ZTZ':
            YY = Y @ Y.T
            fcn = lambda x: likelihood_diagYYT_ZTZ(x, Z, YY, P, self.components, X, self.noise_model, return_deriv=2)
        elif like_fcn == 'YTY_ZZT':
            fcn = lambda x: likelihood_diagYTY_ZZT(x, Z, Y, self.components, X, self.noise_model, return_deriv=2)
        elif like_fcn == 'YTY_ZTZ':
            fcn = lambda x: likelihood_diagYTY_ZTZ(x, Z, Y, self.components, X, self.noise_model, return_deriv=2)
        else:
            raise NameError('like_fcn needs to be auto, YYT_ZZT,....')
        self.theta_, self.trainLogLike_, self.optim_info = pcm.optimize.newton(self.theta0_, fcn, **optim_param)
        return self

    def fit(self, Z ,Y , X = None):
        """
        Estimates the regression parameters, given a specific regularization
        Parameters:
            Z (2d-np.array)
                Design matrix for random effects NxQ
            Y  (2d-np.array)
                NxP Matrix of data
            X (np.array)
                Fixed effects design matrix - will be accounted for by ReML
        Returns:
            self
                Model with fitted parameters

        """
        if (self.theta is None):
            raise NameError('Regularisation parameters (theta) need to be optimized with optimize_regulularization or set')
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
        """
        Predicts new data based on a fitted model
        Parameters:
            Z (2d-np.array)
                Design matrix for random effects NxQ
            Y  (2d-np.array)
                NxP Matrix of data
            X (np.array)
                Fixed effects design matrix - will be accounted for by ReML
        Returns:
            self
                Model with fitted parameters

        """
        if  not hasattr(self,'coef_'):
            raise NameError('Model needs to first be fitted')
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