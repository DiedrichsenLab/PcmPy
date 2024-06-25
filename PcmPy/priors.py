import numpy as np
from numpy import exp, tanh, log, sqrt
import warnings

def normal(theta,mean=0,prec=1):
    """
    Normal prior
    """
    logprior = -0.5 * ((theta-mean)**2  * prec) # independent Gaussian prior
    dlogprior = -(theta - mean) * prec
    ddlogprior = -prec
    return logprior, dlogprior, ddlogprior

def corr_ref(theta,weight=1):
    """
    Reference prior for correlation in fisher z-space
    """
    z= theta
    ez = np.exp(2*z) + 2 + exp(-2*z)
    logprior = weight * log(16)-2*log(ez)
    dlogprior = -weight *4*(exp(2*z)-exp(-2*z))/ez
    ddlogprior = -weight *16/ez
    return logprior,dlogprior,ddlogprior

def corr_flat(theta):
    """
    Flat prior in r space for correlation in fisher z-space = (1-r**2)
    """
    z= theta
    r2 = 1-tanh(z)**2
    logprior = log(r2)
    dlogprior = -2*tanh(z)
    ddlogprior = -2*r2
    return logprior,dlogprior,ddlogprior