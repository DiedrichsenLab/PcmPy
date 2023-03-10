import numpy as np
from numpy import exp, eye, log, sqrt
from numpy.linalg import solve, eigh, cholesky, pinv
import PcmPy as pcm
import pandas as pd

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
    

    def set_theta0(self,G_hat):
        self.theta0 = np.ones((self.n_param,))


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
    it models the correlation between different items  across 2 experimental conditions. Using this paramaterization:
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
            with np.errstate(all='ignore'):
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
    
    def set_theta0(self, Y, Z, X=None):
        """Makes an initial guess on noise parameters
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
            raise(NameError("Too many model factors to estimate noise variance. Consider removing terms or setting fixedEffect to 'none'"))
        self.theta0 = np.array([-1,log(noise0)])


class ModelFamily:
    """
    ModelFamily class is basically a list (iterable) of models,
    which is constructed from a combining a set of components in
    every possible way. Every components can be either switched in or out.
    You can specify a list of 'base components', which are always present.
    A Model family can be either constructed from a component model, or
    a list of (usually fixed) models.
    """
    def __init__(self,components,basecomponents=None,comp_names=None):
        """
        Args:
            components (list)
                A list of model components, which are used to create the model family
                Can be a list of fixed models, a component model, or a num_comp x N xN array
            basecomponents (list)
                This specifies the components that are present everywhere
        """
        if type(components) is ComponentModel:
            self.num_comp = components.Gc.shape[0]
            self.Gc = components.Gc
            self.comp_names = comp_names
        elif type(components) is np.ndarray:
            if components.ndim != 3:
                raise(NameError('ndarray input needs to have 3 dimensions (num_comp x N x N'))
            self.num_comp = components.shape[0]
            self.Gc = components
            self.comp_names = comp_names
        elif type(components) is list:
            self.num_comp = len(components)
            for i,m in enumerate(components):
                if type(m) is not FixedModel:
                    raise(NameError('Can only construct a model class from fixed models'))
                if i==0:
                    self.Gc=np.empty((len(components),m.G.shape[0],m.G.shape[1]))
                    self.comp_names=np.empty((len(components),),dtype=object)

                self.Gc[i,:,:]=m.G
                self.comp_names[i]=m.name
        else:
            raise(NameError('Input needs to be Component model, ndarray, or a list of fixed models'))

        # concatenate basecomponents to the end of the self.Gc
        if basecomponents is not None:
            if type(basecomponents) is ComponentModel:
                self.num_basecomp = basecomponents.Gc.shape[0]
                self.Gc = np.r_[self.Gc,basecomponents.Gc]
            elif type(basecomponents) is np.ndarray:
                if basecomponents.ndim != 3:
                    raise(NameError('ndarray input needs to have 3 dimensions (num_basecomp x N x N'))
                self.num_basecomp = basecomponents.shape[0]
                self.Gc = np.r_[self.Gc,basecomponents]
            elif type(basecomponents) is list:
                self.num_basecomp = len(basecomponents)
                for i,m in enumerate(basecomponents):
                    if type(m) is not FixedModel:
                        raise(NameError('Can only construct a model class from fixed models'))
                    self.Gc = np.r_[self.Gc,m.G]
            else:
                raise(NameError('Input needs to be Component model, ndarray, or a list of fixed models'))
        else:
            self.num_basecomp = 0

        # Check if component names are given:
        if self.comp_names is None:
            self.comp_names = [f'{d}' for d in np.arange(self.num_comp)+1]
        self.comp_names = np.array(self.comp_names)

        # Build all combination of 0,1,2... components
        if self.num_comp > 12:
            raise(NameError('More than 12 components is probably not recommended '))
        self.num_models = 2 ** self.num_comp
        self.combinations = np.empty((self.num_models,0),dtype=int)

        mn = np.arange(self.num_models)
        for i in range(self.num_comp):
            self.combinations = np.c_[self.combinations,np.floor(mn/(2**i))%2]

        # Order the combinations by the number of components that they contain
        self.num_comp_per_m = self.combinations.sum(axis=1).astype(int)
        ind = np.argsort(self.num_comp_per_m+mn/self.num_models)

        self.num_comp_per_m = self.num_comp_per_m[ind]+self.num_basecomp
        self.combinations = self.combinations[ind,:]

        # Now build all model combinations as individual models
        self.models = []
        self.model_names = []
        for m in range(self.num_models):
            ind = np.r_[self.combinations[m],np.ones(self.num_basecomp)]>0
            if ind.sum()==self.num_basecomp:
                name = 'base'
                if self.num_basecomp==0:
                    mod = FixedModel(name,np.zeros(self.Gc[0].shape))
                else:
                    mod = ComponentModel(name,self.Gc[ind,:,:])
            else:
                name = '+'.join(self.comp_names[self.combinations[m]>0])
                mod = ComponentModel(name,self.Gc[ind,:,:])
            self.model_names.append(name)
            self.models.append(mod)

    def __getitem__(self,key):
        return self.models[key]

    def __len__(self):
        return self.num_models

    def get_layout(self):
        """generate 2d layout of the model tree
        root model will be at (0,0)
        Args:
            None
        Returns:
            x (ndarray):
                x-coordinate of model
            y (ndarray):
                y-coordinate of model
        """
        x = np.zeros((self.num_models,))
        y = np.zeros((self.num_models,))
        max_comp=np.max(self.num_comp_per_m)
        for i in range(max_comp+1):
            ind = self.num_comp_per_m==i
            y[ind]=i
            x_coord = np.arange(np.sum(ind))
            x[ind]= x_coord - x_coord.mean()
        return x,y

    def get_connectivity(self,by_comp=False):
        """ return a connectivty
        matrix that determines whether
        2 models only differ by a single component
        Args:
            by_comp (bool):
                If true, returns (num_model x num_model x num_comp) array
        Returns:
            connect (ndarray):
                0: not connected 1: component added -1: component subtracted
        """
        diff = np.zeros((self.num_comp,self.num_models,self.num_models),dtype=int)
        for i in range(self.num_comp):
            diff[i] = self.combinations[:,i].reshape((-1,1)) - self.combinations[:,i].reshape((1,-1))
        connect = np.sum(np.abs(diff),axis=0)
        diff[:,connect!=1]=0
        if by_comp:
            return diff
        else:
            return np.sum(diff,axis=0)

    def model_posterior(self,likelihood,method='AIC',format='ndarray'):
        """ Determine posterior of the model across model family

        Args:
            likelihood ([np.array or DataFrame]):
                N x num_models log-likelihoods
            method (string):
                Method by which to correct for number of parameters(k)
                'AIC' (default): LL-k
                None: No correction - use if crossvalidated likelihood is used
            format (string):
                Return format for posterior
                'ndarray': Simple N x num_models np.array
                'DataFrame': pandas Data frame
        Returns:
            posterior (DataFrame or ndarray):
                Model posterior - rows are data set, columns are models
        """
        if type(likelihood) in [pd.Series,pd.DataFrame]:
            LL = likelihood.to_numpy()
        else:
            LL = likelihood

        if LL.ndim==1:
            LL=LL.reshape((1,-1))

        if method=='AIC':
            crit = LL - self.num_comp_per_m
        elif method is None:
            crit = LL
        else:
            raise(NameError('Method needs be either BIC, AIC, or None'))

        # Safe transform into probability
        crit = crit - crit.max(axis=1).reshape(-1,1)
        crit = np.exp(crit)
        p = crit / crit.sum(axis=1).reshape(-1,1)

        if format == 'DataFrame':
            return pd.DataFrame(data=p,
                        index=np.arange(p.shape[0]),
                        columns = self.model_names)
        else:
            return p

    def component_posterior(self,likelihood,method='AIC',format='ndarray'):
        """ Determine the posterior of the component (absence / presence)

        Args:
            likelihood ([np.array or DataFrame]):
                N x num_models log-likelihoods
            method (string):
                Method by which to correct for number of parameters(k)
                'AIC' (default): LL-k
                None: No correction - use if crossvalidated likelihood is used
            format (string):
                Return format for posterior
                'ndarray': Simple N x num_models np.array
                'DataFrame': pandas Data frame

        Returns:
            posterior (DataFrame):
                Component posterior - rows are data set, columns are components
        """
        mposterior = self.model_posterior(likelihood,method)
        cposterior = np.empty((mposterior.shape[0],self.num_comp))

        for i in range(self.num_comp):
            cposterior[:,i] = mposterior[:,self.combinations[:,i]==1].sum(axis=1)

        if format == 'DataFrame':
            return pd.DataFrame(data=cposterior,
                        index=np.arange(cposterior.shape[0]),
                        columns = self.comp_names)

        return cposterior

    def component_bayesfactor(self,likelihood,method='AIC',format='ndarray'):
        """ Returns a log-bayes factor for each component

        Args:
            likelihood ([np.array or DataFrame]):
                N x num_models log-likelihoods
            method (string):
                Method by which to correct for number of parameters(k)
                'AIC' (default): LL-k
                None: No correction - use if crossvalidated likelihood is used
            format (string):
                Return format for posterior
                'ndarray': Simple N x num_models np.array
                'DataFrame': pandas Data frame

        Returns:
            posterior (DataFrame):
                Component posterior - rows are data set, columns are components
        """
        mposterior = self.model_posterior(likelihood,method)
        c_bf = np.empty((mposterior.shape[0],self.num_comp))

        for i in range(self.num_comp):
            c_bf[:,i] = np.log(mposterior[:,self.combinations[:,i]==1].sum(axis=1))-np.log(mposterior[:,self.combinations[:,i]==0].sum(axis=1))

        if format == 'DataFrame':
            return pd.DataFrame(data=c_bf,
                        index=np.arange(c_bf.shape[0]),
                        columns = self.comp_names)
        return c_bf

    def component_exptheta(self,theta):
        if len(theta)!=self.num_models:
            raise(NameError(f'Length of theta: {len(theta)}, Number of models: {self.num_models} must match'))
        num_sub = theta[0].shape[1]
        exptheta = np.zeros((self.num_comp,num_sub,self.num_models))
        for m in range(self.num_models):
            indx = np.nonzero(self.combinations[m])[0]
            for j,ind in enumerate(indx):
                exptheta[ind,:,m]=np.exp(theta[m][j,:])
        return exptheta

    def component_varestimate(self,
            likelihood,
            theta,
            method='AIC',
            format='ndarray'):
        """ Returns a log-bayes factor for each component

        Args:
            likelihood (np.array or DataFrame):
                N x num_models log-likelihoods
            theta (np.array):
                N x num_models of fitted parameters
            method (string):
                Method by which to correct for number of parameters(k)
                'AIC' (default): LL-k
                None: No correction - use if crossvalidated likelihood is used
            format (string):
                Return format for posterior
                'ndarray': Simple N x num_models np.array
                'DataFrame': pandas Data frame

        Returns:
            posterior (DataFrame):
                Component posterior - rows are data set, columns are components
        """
        mposterior = self.model_posterior(likelihood,method)
        exptheta = self.component_exptheta(theta)

        var_est = np.sum(exptheta * mposterior,axis=2).T
        var_comp = np.trace(self.Gc,axis1=1,axis2=2)
        var_est = var_est * var_comp

        if format == 'DataFrame':
            return pd.DataFrame(data=var_est,
                        index=np.arange(var_est.shape[0]),
                        columns = self.comp_names)
        return var_est
