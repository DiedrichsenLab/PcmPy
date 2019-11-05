"""
Created on Sun Oct 13 20:19:10 2019

@author: jdiedrichsen
"""

import PcmPy as pcm 
import numpy as np 
import scipy.stats as ss 


import scipy.linalg as sl 

class Dataset: 
    """
    Dataset class contains one data set - or multiple data sets with the same structure
    """
    def __init__(self,measurements,descriptors = None ,obs_descriptors = None,channel_descriptors = None): 
        """
        Creator for Dataset class
        
        Args: 
            measurements (numpy.ndarray):   n_obs x n_channel 2d-array, or n_set x n_obs x n_channel 3d-array 
            descriptors (dict):             descriptors with 1 value per Dataset object 
            obs_descriptors (dict):         observation descriptors (all are array-like with shape = (n_obs,...)) 
            channel_descriptors (dict):     channel descriptors (all are array-like with shape = (n_channel,...))
        Returns: 
            dataset object 
        """
        if (measurements.ndim==2):
            self.measurements = measurements
            self.n_set = 1 
            self.n_obs,self.n_channel = self.measurements.shape
        elif (measurements.ndim==3):
            self.measurements = measurements
            self.n_set,self.n_obs,self.n_channel = self.measurements.shape
        self.descriptors = descriptors 
        self.obs_descriptors = obs_descriptors 
        self.channel_descriptors = channel_descriptors 

def make_design(n_cond,n_part):
    """
    Makes simple fMRI design with n_cond, each measures n_part times 

    Args:
        n_cond (int):          Number of conditions 
        n_part (int):          Number of partitions 
    Returns:
        Tuple (cond_vec,part_vec)
        cond_vec (np.ndarray): n_obs vector with condition 
        part_vec (np.ndarray): n_obs vector with partition 
    """
    p = np.array(range(0,n_part))
    c = np.array(range(0,n_cond))
    cond_vec = np.kron(np.ones((n_part,)),c) # Condition Vector 
    part_vec = np.kron(p,np.ones((n_cond,))) # Partition vector 
    return(cond_vec,part_vec)

def make_dataset(model,theta,cond_vec,n_channel=30,n_sim = 1,\
                 signal = 1,noise = 1,noise_cov = None,\
                 part_vec=None): 
    """
    Simulates a fMRI-style data set with a set of partitions 

    Args:
        model (pcm.Model):        the model from which to generate data 
        theta (numpy.ndarray):    vector of parameters (one dimensional)
        cond_vec (numpy.ndarray): RSA-style model: vector of experimental conditions 
                                  Encoding-style model: design matrix (n_obs x n_cond)
        n_channel (int):          Number of channels (default = 30)
        n_sim (int):              Number of data set simulation with the same signal (default = 1)
        signal (float):           Signal variance (multiplied by predicted G)
        noise (float)             Noise variance (mulitplied by spatial noise_cov if given)
        noise_cov (numpy.ndarray):n_channel x n_channel covariance structure of noise (default = identity)
        part_vec (numpy.ndarray): optional partition vector if within-partition covariance is specified 
    Returns:
        data (pcm.Dataset):       Dataset with obs_descriptors. 
    """

    G,dG = model.calculate_G(theta)    # Get the model prediction 
    
    # Make design matrix 
    if (cond_vec.ndim == 1):
        Zcond = pcm.indicator.identity(cond_vec)
    elif (cond_vec.ndim == 2): 
        Zcond = cond_vec
    else:
        raise(NameError("cond_vec needs to be either condition vector or design matrix"))
    n_obs,n_cond = Zcond.shape
    
    # If noise_cov given, precalculate the cholinsky decomp 
    if (noise_cov is not None): 
        if (noise_cov.shape is not (n_channel,n_channel)):
            raise(NameError("noise covariance needs to be n_channel x n_channel array"))
        noise_chol = np.linalg.cholesky(noise_cov)
        
    # Generate the true patterns with exactly correct second moment matrix 
    true_U = np.random.uniform(0,1,size=(n_cond,n_channel))
    true_U = ss.norm.ppf(true_U)  # We use two-step procedure allow for different distributions later on 
    # Make orthonormal row vectors  
    E = true_U @ true_U.transpose() 
    L = np.linalg.cholesky(E)   
    true_U = np.linalg.solve(L,true_U)
    
    # Now produce data with the known second-moment matrix 
    # Use positive eigenvectors only (cholesky does not work with rank-deficient matrices)
    l,V=np.linalg.eig(G)  
    l[l<1e-15]=0          
    l = np.sqrt(l)   
    chol_G = V.real*l.real.reshape((1,l.size))  
    true_U = (chol_G @ true_U) * np.sqrt(n_channel)
    
    # Generate noise as a matrix normal, independent across partitions
    # If noise covariance structure is given, it is assumed that it's the same 
    # across different partitions 
    data = np.empty((n_sim,n_obs,n_channel))
    for i in range(0,n_sim):
        epsilon = np.random.uniform(0,1,size=(n_obs,n_channel))
        epsilon = ss.norm.ppf(epsilon)*np.sqrt(noise)  # Allows alter for providing own cdf for noise distribution 
        if (noise_cov is not None):         
            epsilon   = epsilon @ noise_chol
        data[i,:,:] = Zcond@true_U * np.sqrt(signal) + epsilon
    obs_des = {"cond_vec": cond_vec}
    des     = {"signal": signal,"noise":noise,"model":model.name,"theta": theta}
    dataset = pcm.Dataset(data,obs_descriptors = obs_des,descriptors = des)
    return dataset
