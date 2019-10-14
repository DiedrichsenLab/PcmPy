"""
Created on Sun Oct 13 20:19:10 2019

@author: jdiedrichsen
"""

import PcmPy as pcm 
import numpy as np 
import scipy.stats as ss 


import scipy.linalg as sl 

class Dataset: 
    
    def __init__(self,measurements,descriptors = {} ,obs_descriptors = {},channel_descriptors = {}): 
        """
        Creator of dataset class, using empty dicts as default 
        """
        self.measurements = measurements
        self.n_obs,self.n_channel = self.measurements.shape
        self.descriptors = descriptors 
        self.obs_descriptors = obs_descriptors 
        self.channel_descriptors = channel_descriptors 

def make_design(n_part,n_cond):
    p = np.array(range(0,n_part))
    c = np.array(range(0,n_cond))
    cond_vec = np.kron(np.ones((n_part,)),c) # Condition Vector 
    part_vec = np.kron(p,np.ones((n_cond,))) # Partition vector 
    return(cond_vec,part_vec)
      
def make_dataset(model,theta,cond_vec,part_vec,n_channel=30,n_sim = 1,signal = 1,noise = 1): 
    """
    Simulates a fMRI-style data set with a set of partitions 
    """
    G,dG = model.calculate_G(theta)    # Get the model prediction 
    if (cond_vec.ndim == 1):
        Zcond = pcm.indicator_matrix("identity",cond_vec)
    elif (cond_vec.ndim == 2): 
        Zcond = cond_vec
    else:
        raise(NameError("cond_vec needs to be either condition vector or design matrix"))
    
    n_obs,n_cond = Zcond.shape
    
    # Generate the true patterns with exactly correct second moment matrix 
    U = np.random.uniform(0,1,size=(n_cond,n_channel))
    U = ss.norm.ppf(U)  # We use two-step procedure allow for different distributions later on 
    E = U @ U.transpose()
    M=np.linalg.cholesky(E)
    
    
    # Make the second-moment Y*Y' exactly equal to G
    
    return Y
