"""
Composite test for modelfamily class
@author: jdiedrichsen
"""

import PcmPy as pcm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from scipy import linalg

def two_by_three_design(orthogonalize=False):
    A = np.array([[1.0,0,0],[1,0,0],[0,1,0],[0,1,0],[0,0,1],[0,0,1]])
    B = np.array([[1.0,0],[0,1],[1,0],[0,1],[1,0],[0,1]])
    I = np.eye(6)
    if orthogonalize:
        A = A - A.mean(axis=0)
        B = B - B.mean(axis=0)
        I = I - I.mean(axis=0)
        X= np.c_[A,B]
        I = I-X @ linalg.pinv(X) @ I
    Gc = np.zeros((3,6,6))
    Gc[0]=A@A.T
    Gc[1]=B@B.T
    Gc[2]=I@I.T
    M = pcm.ComponentModel('A+B+I',Gc)
    MF=pcm.model.ModelFamily(M,comp_names=['A','B','I'])
    return M,MF

def component_inference(D,MF):
    T,theta=pcm.fit_model_individ(D,MF,verbose=False)

    # pcm.vis.model_plot(T.likelihood-MF.num_comp_per_m)
    mposterior = MF.model_posterior(T.likelihood.mean(axis=0),method='AIC',format='DataFrame')
    cposterior = MF.component_posterior(T.likelihood,method='AIC',format='DataFrame')
    c_bf = MF.component_bayesfactor(T.likelihood,method='AIC',format='DataFrame')
    var_est = MF.component_varestimate(T.likelihood,theta,method='AIC',format='DataFrame')

    fig=plt.figure(figsize=(10,10))
    plt.subplot(2,2,1)
    pcm.vis.plot_tree(MF,mposterior,show_labels=False,show_edges=True)
    ax=plt.subplot(2,2,2)
    pcm.vis.plot_component(cposterior,type='posterior')
    ax=plt.subplot(2,2,3)
    pcm.vis.plot_component(c_bf,type='bf')
    ax=plt.subplot(2,2,4)
    pcm.vis.plot_component(var_est,type='varestimate')


def sim_two_by_three(theta):
    """Simulates a simple 2x3 factorial design
    Using non-orthogonalized contrasts
    """
    M,MF1 = two_by_three_design(orthogonalize=False)
    M,MF2 = two_by_three_design(orthogonalize=True)
    [cond_vec,part_vec]=pcm.sim.make_design(6,8)
    D = pcm.sim.make_dataset(M,theta,
                            signal=0.1,
                            n_sim = 20,
                            n_channel=20,part_vec=part_vec,
                            cond_vec=cond_vec)
    component_inference(D,MF1)
    component_inference(D,MF2)
    pass

def random_design(N=10,Q=5,num_feat=2,seed=1):
    Gc = np.empty((Q,N,N))
    rng = np.random.default_rng(seed)
    for q in range(Q):
        X= rng.normal(0,1,(N,num_feat))
        X = X/np.sqrt(np.sum(X**2,axis=0)) 
        Gc[q,:,:]= X @ X.T
    M = pcm.ComponentModel('A+B+I',Gc)
    MF=pcm.model.ModelFamily(Gc)
    return M,MF

def random_example(theta,N=10):
    Q = theta.shape[0]
    M,MF = random_design(N=N,Q=Q)
    for q in range(Q):
        plt.subplot(1,Q,q+1)
        plt.imshow(M.Gc[q,:,:])

    [cond_vec,part_vec]=pcm.sim.make_design(N,8)
    D = pcm.sim.make_dataset(M,theta,
                            signal=0.1,
                            n_sim = 20,
                            n_channel=20,part_vec=part_vec,
                            cond_vec=cond_vec)
    component_inference(D,MF)
    pass


def colinear_example(theta,N=20,signal=0.1,seed=1):
    Q = theta.shape[0]
    Gc = np.empty((Q,N,N))
    rng = np.random.default_rng(seed)
    
    Z=rng.normal(0,1,(N,Q))
    X = Z.copy() 
    X[:,0]=0.55*Z[:,0]+0.5*Z[:,1]
    X[:,1]=0.5*Z[:,0]+0.55*Z[:,1]
    X[:,3]=0.3*Z[:,3]+0.7*Z[:,4]
    X[:,4]=0.7*Z[:,3]+0.3*Z[:,4]

    X = X/np.sqrt(np.sum(X**2,axis=0)) 
    
    for q in range(Q):
        Gc[q,:,:]= X[:,q].reshape((N,1)) @ X[:,q].reshape((1,N))
    M = pcm.ComponentModel('Full',Gc)
    MF=pcm.model.ModelFamily(Gc)

    
    for q in range(Q):
        plt.subplot(1,Q,q+1)
        plt.imshow(M.Gc[q,:,:])

    [cond_vec,part_vec]=pcm.sim.make_design(N,8)
    D = pcm.sim.make_dataset(M,theta,
                            signal=signal,
                            n_sim = 20,
                            n_channel=20,part_vec=part_vec,
                            cond_vec=cond_vec)
    component_inference(D,MF)
    pass


if __name__ == '__main__':
    # sim_two_by_three(np.array([-1.0,-1.0,-1.0]))
    colinear_example(np.array([0,-np.inf,-np.inf,-np.inf,-np.inf]),signal=1)
