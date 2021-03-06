import numpy as np
from numpy import exp, eye, log
from numpy.linalg import solve, eigh, cholesky, pinv
import PcmPy as pcm

class Model:
    """
    Abstract PCM Model Class
    """
    def __init__(self,name):
        self.name = name
        self.n_param = 0
        self.algorithm = 'newton' # Default optimization algorithm
        self.theta0 = np.zeros((0,)) # Empty theta0

    def predict(self,theta):
        raise(NameError("caluclate G needs to be implemented"))

    def set_theta0(self,G_hat):
        pass

class ModelFeature(Model):
    """
    Feature model
    A = sum (theta_i *  Ac_i)
    G = A*A'
    """
    def __init__(self,name,Ac):
        """
        Creator for ModelFeature class

        Args:
            name (string):     name of the particular model for indentification
            Ac (numpy.ndarray): 3-dimensional array with components of A
        Returns:
            Model object
        """

        Model.__init__(self,name)
        if (Ac.ndim <3):
            Ac = Ac.reshape((1,)+Ac.shape)
        self.Ac = Ac
        self.n_param = Ac.shape[0]

    def predict(self,theta):
        """
        Calculation of G

        Args:
            theta (numpy.ndarray)
                Vector of model parameters
        Returns:
            G (np.ndarray)
                2-dimensional (K,K) array of predicted second moment
            dG_dTheta (np.ndarray)
                3-d (n_param,K,K) array of partial matrix derivatives of G in respect to theta
        """

        Ac = self.Ac * np.reshape(theta,(theta.size,1,1))# Using Broadcasting
        A = Ac.sum(axis=0)
        G = A@A.transpose()
        dG_dTheta = np.zeros((self.n_param,)+G.shape)
        for i in range(0,self.n_param):
            dA = self.Ac[i,:,:] @ A.transpose()
            dG_dTheta[i,:,:] =  dA + dA.transpose()
        return (G,dG_dTheta)

class ModelComponent(Model):
    """
    Component model class
    G = sum (exp(theta_i) * Gc_i)
    """
    def __init__(self,name,Gc):
        """
        Creator for ModelComponent class

        Parameters:
            name (string)
                name of the particular model for indentification
            Gc (numpy.ndarray)
                3-dimensional array with compoments of G
        Returns:
            Model object
        """
        Model.__init__(self,name)
        if type(Gc) is list:
            Gc = np.stack(Gc,axis=0)
        if (Gc.ndim <3):
            Gc = Gc.reshape((1,)+Gc.shape)
        self.Gc = Gc
        self.n_param = Gc.shape[0]

    def predict(self,theta):
        """
        Calculation of G

        Args:
            theta (numpy.ndarray):    Vector of model parameters
        Returns:
            G (np.ndarray)
                2-dimensional (K,K) array of predicted second moment
            dG_dTheta (np.ndarray)
                3-d (n_param,K,K) array of partial matrix derivatives of G in respect to theta
        """

        exp_theta=np.reshape(np.exp(theta),(theta.size,1,1)) # Bring into the right shape for broadcasting
        dG_dTheta = self.Gc * exp_theta  # This is also the derivative dexp(x)/dx = exp(x)
        G = dG_dTheta.sum(axis=0)
        return (G,dG_dTheta)

    def set_theta0(self,G_hat):
        """
        Sets theta0 based on the crossvalidated second-moment

        Parameters:
            G_hat (numpy.ndarray)
                Crossvalidated estimate of G
        """
        if self.n_param==0:
            self.theta0 = np.zeros((0,))
        else:
            X = np.zeros((G_hat.shape[0]**2, self.n_param))
            for i in range(self.n_param):
                X[:,i] = self.Gc[i,:,:].reshape((-1,))
            h0 = pinv(X) @ G_hat.reshape((-1,1))
            h0[h0<10e-4] = 10e-4
            self.theta0 = log(h0.reshape(-1,))

class ModelFixed(Model):
    """
    Fixed PCM with a rigid predicted G matrix and no parameters
    """
    def __init__(self,name,G):
        """
        Creator for ModelFixed class

        Parameters:
            name (string)
                name of the particular model for indentification
            G (numpy.ndarray)
                2-dimensional array giving the predicted second moment
        Returns:
            Model object
        """

        Model.__init__(self,name)
        if (G.ndim>2):
            raise(NameError("G-matrix needs to be 2-d array"))
        self.G = G
        self.n_param = 0

    def predict(self,theta=None):
        """
        Calculation of G

        Returns:
            G (np.ndarray)
                2-dimensional (K,K) array of predicted second moment
            dG_dTheta
                None
        """

        return (self.G,None)

class ModelFree(Model):
    """
    Free model class: Second moment matrix is 
    G = A*A', where A is a upper triangular matrix that is flexible
    """
    def __init__(self,name,n_cond):
        """
        Creator for ModelFree class

        Parameters:
            name (string)
                name of the particular model for indentification
            n_cond (int)
                number of conditions for free model
        Returns:
            Model object
        """
        Model.__init__(self,name)
        self.n_cond = n_cond
        self.index = np.tri(n_cond)
        self.row, self.col = np.where(self.index)
        self.n_param = len(self.row)

    def predict(self,theta):
        """
        Calculation of G

        Args:
            theta (numpy.ndarray):    Vector of model parameters
        Returns:
            G (np.ndarray)
                2-dimensional (K,K) array of predicted second moment
            dG_dTheta (np.ndarray)
                3-d (n_param,K,K) array of partial matrix derivatives of G in respect to theta
        """

        A = np.zeros((self.n_cond, self.n_cond))
        A[self.row, self.col] = theta
        G = A @ A.T
        dGdtheta = np.zeros((self.n_param,self.n_cond,self.n_cond))
        for i in range (self.n_param):
            dGdtheta[i,self.row[i],:] += A[:,self.col[i]]
            dGdtheta[i,:,self.row[i]] += A[:,self.col[i]]
        return (G,dGdtheta)

    def set_theta0(self,G_hat):
        """
        Sets theta0 based on the crossvalidated second-moment

        Parameters:
            G_hat (numpy.ndarray)
                Crossvalidated estimate of G
        """
        G_pd = pcm.util.make_pd(G_hat)
        A   = cholesky(G_pd) 
        self.theta0 = A[self.row, self.col]


class NoiseModel:
    """
    Abstract PCM Noise model class
    """
    def __init__(self):
        pass

    def predict(self, theta):
        raise(NameError("predict needs to be implemented"))

    def inverse(self, theta):
        raise(NameError("inverse needs to be implemented"))

    def derivative(self, theta):
        raise(NameError("derivative needs to be implemented"))

    def set_theta0(self, Y, Z, X=None):
        raise(NameError("get_theta0 needs to be implemented"))

class IndependentNoise(NoiseModel):
    def __init__(self):
        """
        Creator for Noise model
        """
        NoiseModel.__init__(self)
        self.n_param = 1
        theta0 = 0

    def predict(self, theta):
        return np.exp(theta[0])

    def inverse(self, theta):
        return 1./np.exp(theta[0])

    def derivative(self, theta, n=0):
        return np.exp(theta[0])

    def set_theta0(self, Y, Z, X=None):
        N, P = Y.shape
        if X is not None:
            Z = np.c_[Z, X]
        RY = Y - Z @ pinv(Z) @ Y
        noise0 = np.sum(RY*RY)/(P * (N - Z.shape[1]))
        if noise0 <= 0:
            raise(NameError("Too many model factors to estimate noise variance. Consider removing terms or setting runEffect to 'none'"))
        self.theta0 = np.array([log(noise0)])

class BlockPlusIndepNoise(NoiseModel):
    def __init__(self,part_vec):
        """
        Creator for ModelFixed class
        """
        NoiseModel.__init__(self)
        self.n_param = 2
        self.part_vec = part_vec
        self.B = pcm.matrix.indicator(part_vec)
        self.N, self.M = self.B.shape
        self.BTB = np.sum(self.B,axis=0)
        self.BBT = self.B @ self.B.T

    def predict(self, theta):
        S = self.BBT * exp(theta[0]) + eye(self.N) * exp(theta[1])
        return S

    def inverse(self, theta):
        sb = exp(theta[0]) # Block parameter
        se = exp(theta[1])
        A = eye(self.M) * se / sb + self.B.T @ self.B
        S = (eye(self.N) - self.B @ np.linalg.solve(A,self.B.T)) / se
        return S

    def derivative(self, theta,n=0):
        if n==0:
            return self.BBT * np.exp(theta[0])
        elif n==1:
            return eye(self.N) * np.exp(theta[1])