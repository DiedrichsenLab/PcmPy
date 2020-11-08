"""
Regression module contains bare-bones version of  the PCM toolbox that can be used to tune ridge/Tikhonov coefficients in the context of tranditional regression models. No assumption are made about independent data partitions.
"""

import numpy as np
from numpy.linalg import solve, eigh, cholesky
from numpy import sum, diag, log, eye, exp, trace, einsum
import PcmPy as pcm

def likelihood_diag(theta, Z, Y, comp, X=None, Noise=pcm.model.IndependentNoise(), return_deriv=0):
    """
    Negative Log-Likelihood of the data and derivative in respect to the parameters

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
    B = Y @ Y.T @ iVr
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


class RidgeDiag:
    """
        Class for Linear Regression with Tikhonov (L2) regularization.
        The regularization matrix for this class is diagnonal, with groups
        of elements along the diagonal sharing the same Regularisation factor.
    """
    def __init__(self, components, theta0 = None, Noise_model, fit_intercept = True):
        self.components = components
        self.n_param = max(components)+2
        self.optimizer = 'newton' # Default optimization algorithm
        self.theta0_ = np.zeros((n_param,)) # Empty theta0
        self.theta_  = np.zeros((n_param,))

    def optimize_lambda(self, Z , Y, X = None):
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
        n_param = max(comp)+1+Noise.n_param
        th0 = np.zeros((n_param,))
        fcn = lambda x: likelihood_diag(x, Z, Y, comp, X, Noise,return_deriv=2)
        theta, l, INFO = pcm.optimize.newton(th0, fcn, **optim_param)
        return theta, l, INFO


    def predict(self,theta):
        raise(NameError("caluclate G needs to be implemented"))

    def set_params(self,params):

    def get_params(self,params):
