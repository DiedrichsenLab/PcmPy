import numpy as np
class Model:
    """
    Abstract PCM Model Class
    """
    def __init__(self,name):
        self.name = name

    def predict(self,theta):
        raise(NameError("caluclate G needs to be implemented"))

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
            dA = self.Ac[i,:,:] @ A.transpose();
            dG_dTheta[i,:,:] =  dA + dA.transpose();
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

class ModelFixed(Model):
    """
    Fixed PCM with a rigid predicted G matrix and no parameters:
    Use for Null-models
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
            raise(NameError("G-matrix needs to be 2-d array"));
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

class Noise:
    """
    Abstract PCM Noise model class
    """
    def __init__(self,name):
        self.name = name

    def predict(self,theta):
        raise(NameError("predict needs to be implemented"))