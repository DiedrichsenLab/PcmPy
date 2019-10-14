import numpy as np 
class Model:
    """
    Abstract Model Class
    """
    def __init__(self,name):
        self.name = name
    
    def calculate_G(self,theta):
        raise(NameError("caluclate G needs to be implemented"))
    
class ModelFeature(Model):
    """
    Feature model 
    A = sum (theta_i *  Ac_i)
    G = A*A' 
    """
    def __init__(self,name,Ac): 
        Model.__init__(self,name)
        if (Ac.ndim <3):
            Ac = Ac.reshape((1,)+Ac.shape)
        self.Ac = Ac 
        self.n_param = Ac.shape[0]
        
    def calculate_G(self,theta):
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
        Model.__init__(self,name)
        if (Gc.ndim <3):
            Gc = Gc.reshape((1,)+Gc.shape)
        self.Gc = Gc 
        self.n_param = Gc.shape[0]
        
    def calculate_G(self,theta):
        exp_theta=np.reshape(np.exp(theta),(theta.size,1,1)) # Bring into the right shape for broadcasting   
        dG_dTheta = self.Gc * exp_theta  # This is also the derivative dexp(x)/dx = exp(x) 
        G = dG_dTheta.sum(axis=0)
        return (G,dG_dTheta)
        
