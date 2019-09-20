import numpy as np 
class PcmModel:
    
    def __init__(self,type,name):
        self.type = type
        self.name = name
    
    def calculateG(self,theta):
        raise(NameError("caluclate G needs to be implemented"))
    
class FeatureModel(PcmModel):
    def __init__(self,name,Ac): 
        PcmModel.__init__(self,name,"feature")
        if (Ac.ndim <3):
            Ac = Ac.reshape((1,)+A.shape)
        self.Ac = Ac 
        self.numGparams = Ac.shape[0]
        
    
class ComponentModel(PcmModel):
    

"""

function [G,dGdtheta] = pcm_calculateG(M,theta)
% function [G,dGdtheta] = pcm_calculateG(M,theta)
% This function calculates the predicted second moment matrix (G) and the
% derivate of the second moment matrix in respect to the parameters theta. 
% INPUT: 
%       M:        Model structure 
%       theta:    Vector of parameters 
% OUTPUT: 
%       G:        Second moment matrix 
%       dGdtheta: Matrix derivatives in respect to parameters 
% Joern Diedrichsen, 2016 
@author: jdiedrichsen
"""
