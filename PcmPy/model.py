import numpy as np
from numpy import exp, eye, log, sqrt
from numpy.linalg import solve, eigh, cholesky, pinv
import PcmPy as pcm

class Model:
    """
        Abstract PCM Model Class
    """
    def __init__(self,name):
        """

        Args:
            name ([str]): Name of the the model
        """
        self.name = name
        self.n_param = 0
        self.algorithm = 'newton' # Default optimization algorithm
        self.theta0 = np.zeros((0,)) # Empty theta0

    def predict(self,theta):
        """
        Prediction function: Needs to be implemented
        """
        raise(NameError("caluclate G needs to be implemented"))

    def set_theta0(self,G_hat):
        pass

class FeatureModel(Model):
    """
    Feature model:
    A = sum (theta_i *  Ac_i)
    G = A*A'
    """
    def __init__(self,name,Ac):
        """

        Args:
            name (string)
                name of the particular model for indentification
            Ac (numpy.ndarray)
                3-dimensional array with components of A
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
            theta (np.ndarray)
                Vector of model parameters
        Returns:
            G (np.ndarray):
                2-dimensional (K,K) array of predicted second moment
            dG_dTheta (np.ndarray):
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

class ComponentModel(Model):
    """
    Component model class
    G = sum (exp(theta_i) * Gc_i)
    """
    def __init__(self,name,Gc):
        """

        Parameters:
            name (string):
                name of the particular model for indentification
            Gc (numpy.ndarray):
                3-dimensional array with compoments of G
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

        Parameters:
            theta (numpy.ndarray):    Vector of model parameters
        Returns:
            G (np.ndarray):
                2-dimensional (K,K) array of predicted second moment
            dG_dTheta (np.ndarray):
                3-d (n_param,K,K) array of partial matrix derivatives of G in respect to theta
        """

        exp_theta=np.reshape(np.exp(theta),(theta.size,1,1)) # Bring into the right shape for broadcasting
        dG_dTheta = self.Gc * exp_theta  # This is also the derivative dexp(x)/dx = exp(x)
        G = dG_dTheta.sum(axis=0)
        return (G,dG_dTheta)

    def set_theta0(self,G_hat):
        """Sets theta0 based on the crossvalidated second-moment

        Parameters:
            G_hat (numpy.ndarray):
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

class CorrelationModel(Model):
    """
    Correlation model class for a fixed or flexible correlation model
    it models the correlation between different items  across 2 experimental conditions.
    In this paramaterization:
    var(x) = exp(theta_x)

    var(y) = exp(theta_y)

    cov(x,y) = sqrt(var(x)*var(y))* r

    r = (exp(2*theta_z)-1)/(exp(2*theta_z)+1);  % Fisher inverse
    """

    def __init__(self,name,within_cov = None,num_items=1,
                corr=None,cond_effect = False):
        """

        Parameters:
            name (string):
                name of the particular model for indentification
            within_cov (numpy.ndarray or None):
                how to model within condition cov-ariance between items
            num_items (int):
                Number of items within each condition
        """
        Model.__init__(self,name)
        self.num_items = num_items
        self.num_cond = 2 # Current default
        self.cond_effect = cond_effect

        self.cond_vec = np.kron(np.arange(self.num_cond), np.ones((self.num_items,)))
        self.item_vec = np.kron(np.ones((self.num_cond,)),np.arange(self.num_items))
        K = self.num_cond * self.num_items

        if within_cov is None:
            self.within_cov = np.eye(num_items).reshape(1,num_items,num_items)
        else:
            self.within_cov = within_cov

        # Initialize the Gc structure
        self.n_cparam = self.num_cond * self.cond_effect # Number of condition effect parameters
        self.n_wparam = self.within_cov.shape[0] # Number of within condition parameters
        self.n_param = self.num_cond * self.n_wparam + self.n_cparam
        self.Gc = np.zeros((self.n_param,K,K))

        # Now add the condition effect and within condition covariance structure
        for i in range(self.num_cond):
            ind = np.where(self.cond_vec==i)[0]
            if self.cond_effect:
                self.Gc[np.ix_([i],ind,ind)] = 1
            c = np.arange(self.n_wparam) + i * self.n_wparam + self.n_cparam
            self.Gc[np.ix_(c,ind,ind)] = self.within_cov

        # Check if fixed or flexible correlation model
        self.corr = corr
        if self.corr is None:
            self.n_param = self.n_param + 1

    def predict(self,theta):
        """
        Calculation of G for a correlation model

        Parameters:
            theta (numpy.ndarray):    Vector of model parameters
        Returns:
            G (np.ndarray)
                2-dimensional (K,K) array of predicted second moment
            dG_dTheta (np.ndarray)
                3-d (n_param,K,K) array of partial matrix derivatives of G in respect to theta
        """
        # Determine the correlation to model
        if self.corr is None:
            z = theta[-1] # Last item
            if np.abs(z)>150:
                r=np.nan
            else:
                r = (exp(2*z)-1)/(exp(2*z)+1) # Fisher inverse transformation
        else:
            r = self.corr

        # Get the basic variances within conditons
        n = self.n_wparam * self.num_cond + self.num_cond * self.cond_effect # Number of basic parameters
        o = self.num_cond * self.cond_effect # Number of condition
        dG_dTheta = np.zeros((self.n_param,self.Gc.shape[1],self.Gc.shape[1]))
        exp_theta=np.reshape(np.exp(theta[0:n]),(n,1,1)) # Bring into the right shape for broadcasting
        dG_dTheta[0:n,:,:] = self.Gc * exp_theta  # This is also the derivative dexp(x)/dx = exp(x)
        # Sum current G-matrix without the condition effects
        G = dG_dTheta[o:n,:,:].sum(axis=0)

        # Now determine the cross_condition block (currently only for 2 conditions)
        i1 = np.where(self.cond_vec==0)[0]
        i2 = np.where(self.cond_vec==1)[0]
        p1 = np.arange(self.n_wparam) + self.n_cparam
        p2 = p1 + self.n_wparam
        C = sqrt(G[np.ix_(i1,i1)] * G[np.ix_(i2,i2)]) # Maximal covariance
        G[np.ix_(i1,i2)] = C * r
        G[np.ix_(i2,i1)] = C.T * r

        # Now add the across-conditions blocks to the derivatives:
        for j in range(self.n_wparam):
            dG1 = dG_dTheta[np.ix_([p1[j]],i1,i1)]
            dG1 = dG1[0,:,:]
            G1 = G[np.ix_(i1,i1)]
            dG2 = dG_dTheta[np.ix_([p2[j]],i2,i2)]
            dG2 = dG2[0,:,:]
            G2 = G[np.ix_(i2,i2)]
            dC1 = np.zeros(dG1.shape)
            dC2 = np.zeros(dG2.shape)
            ind = C!=0
            dC1[ind] = 0.5 * 1/C[ind] * r * G2[ind] * dG1[ind]
            dC2[ind] = 0.5 * 1/C[ind] * r * G1[ind] * dG2[ind]
            dG_dTheta[np.ix_([p1[j]],i1,i2)] = dC1
            dG_dTheta[np.ix_([p1[j]],i2,i1)]=  dC1.T
            dG_dTheta[np.ix_([p2[j]],i1,i2)] = dC2
            dG_dTheta[np.ix_([p2[j]],i2,i1)] = dC2.T

        # Now add the main  Condition effect co-variance
        G = G+dG_dTheta[0:o,:,:].sum(axis=0)

        # Add the derivative for the correlation parameter for flexible models
        if self.corr is None:
            dC = C*4*exp(2*z)/(exp(2*z)+1)**2
            dG_dTheta[np.ix_([n],i1,i2)] = dC
            dG_dTheta[np.ix_([n],i2,i1)] = dC.T
        return (G,dG_dTheta)

    def set_theta0(self,G_hat):
        """
        Sets theta0 based on the crossvalidated second-moment

        Parameters:
            G_hat (numpy.ndarray)
                Crossvalidated estimate of G
        """
        n_p = self.n_param - (self.corr is None)
        G_hat = pcm.util.make_pd(G_hat)
        X = np.zeros((G_hat.shape[0]**2, n_p))
        for i in range(n_p):
            X[:,i] = self.Gc[i,:,:].reshape((-1,))
        h0 = pinv(X) @ G_hat.reshape((-1,1))
        h0[h0<10e-4] = 10e-4
        self.theta0 = log(h0.reshape(-1,))
        if self.corr is None:
            self.theta0 = np.concatenate([self.theta0,np.zeros((1,))])

    def get_correlation(self,theta):
        """
        Returns the correlations from a set of fitted parameters

        Parameters:
            theta (numpy.ndarray):
                n_param vector or n_param x n_subj matrix of model parameters

        Returns:
            correlations (numpy.ndarray)
                Correlation value
        """
        N , n_param = theta.shape
        if self.corr is None:
            z = theta[self.n_param-1]
            r = (exp(2*z)-1)/(exp(2*z)+1)
        else:
            r = self.corr # Fixed correlations
        return r

class FixedModel(Model):
    """
    Fixed PCM with a rigid predicted G matrix and no parameters
    """
    def __init__(self,name,G):
        """

        Parameters:
            name (string)
                name of the particular model for indentification
            G (numpy.ndarray)
                2-dimensional array giving the predicted second moment
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
            dG_dTheta (None)
        """

        return (self.G,None)

class FreeModel(Model):
    """
    Free model class: Second moment matrix is
    G = A*A', where A is a upper triangular matrix that is flexible
    """
    def __init__(self,name,n_cond):
        """

        Parameters:
            name (string)
                name of the particular model for indentification
            n_cond (int)
                number of conditions for free model
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
    """
    Simple Indepdennt noise model (i.i.d)
    the only parameter is the noise variance
    """
    def __init__(self):
        NoiseModel.__init__(self)
        self.n_param = 1
        theta0 = 0

    def predict(self, theta):
        """
        Prediction function returns S - predicted noise covariance matrix

        Args:
            theta ([np.array]): Array like of noiseparamters

        Returns:
            s (double)
                Noise variance (for simplicity as a scalar)
        """
        return np.exp(theta[0])

    def inverse(self, theta):
        """
        Returns S^{-1}

        Args:
            theta ([np.array]): Array like of noiseparamters

        Returns:
            s (double)
                Inverse of noise variance (scalar)
        """
        return 1./np.exp(theta[0])

    def derivative(self, theta, n=0):
        """
        Returns the derivative of S in respect to it's own parameters

        Args:
            theta ([np.array])
                Array like of noiseparamters
            n (int, optional)
                Number of parameter to get derivate for. Defaults to 0.

        Returns:
            d (np-array)
                derivative of S in respective to theta
        """
        return np.exp(theta[0])

    def set_theta0(self, Y, Z, X=None):
        """Makes an initial guess on noise paramters

        Args:
            Y ([np.array])
                Data
            Z ([np.array])
                Random Effects matrix
            X ([np.array], optional)
                Fixed effects matrix.
        """
        N, P = Y.shape
        if X is not None:
            Z = np.c_[Z, X]
        RY = Y - Z @ pinv(Z) @ Y
        noise0 = np.sum(RY*RY)/(P * (N - Z.shape[1]))
        if noise0 <= 0:
            raise(NameError("Too many model factors to estimate noise variance. Consider removing terms or setting runEffect to 'none'"))
        self.theta0 = np.array([log(noise0)])

class BlockPlusIndepNoise(NoiseModel):
    """
    This noise model uses correlated noise per partition (block)
    plus independent noise per observation
    For beta-values from an fMRI analysis, this is an adequate model
    """

    def __init__(self,part_vec):
        """
        Args:
            part_vec ([np.array]): vector indicating the block membership for each observation
        """
        NoiseModel.__init__(self)
        self.n_param = 2
        self.part_vec = part_vec
        self.B = pcm.matrix.indicator(part_vec)
        self.N, self.M = self.B.shape
        self.BTB = np.sum(self.B,axis=0)
        self.BBT = self.B @ self.B.T

    def predict(self, theta):
        """Prediction function returns S - predicted noise covariance matrix

        Args:
            theta ([np.array]): Array like of noiseparamters

        Returns:
            s (np.array):
                Noise covariance matrix
        """

        S = self.BBT * exp(theta[0]) + eye(self.N) * exp(theta[1])
        return S

    def inverse(self, theta):
        """Returns S^{-1}

        Args:
            theta (np.array): Array like of noiseparamters

        Returns:
            iS (np.array): Inverse of noise covariance
        """
        sb = exp(theta[0]) # Block parameter
        se = exp(theta[1])
        A = eye(self.M) * se / sb + self.B.T @ self.B
        S = (eye(self.N) - self.B @ np.linalg.solve(A,self.B.T)) / se
        return S

    def derivative(self, theta,n=0):
        """Returns the derivative of S in respect to it's own parameters

        Args:
            theta (np.array): Array like of noiseparamters
            n (int, optional): Number of parameter to get derivate for. Defaults to 0.

        Returns:
            d (np.array): derivative of S in respective to theta
        """
        if n==0:
            return self.BBT * np.exp(theta[0])
        elif n==1:
            return eye(self.N) * np.exp(theta[1])