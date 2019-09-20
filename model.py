import numpy as np 
class model:
    
    def __init__(self,type):
        self.type = type
        self.numGparams = 0 
        if (type=="fixed"):
            self.numGparams = 0 
        else:
            self.Gc = np.zeros(5,5)

    
        

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
