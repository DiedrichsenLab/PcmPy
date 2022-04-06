"""
Composite test for modelfamily class
@author: jdiedrichsen
"""

import PcmPy as pcm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb


def sim_two_by_three():
    """Simulates a simple 2x3 factorial design
    """
    M = []
    A = np.array([[1.0,0,0],[1,0,0],[0,1,0],[0,1,0],[0,0,1],[0,0,1]])
    B = np.array([[1.0,0],[0,1],[1,0],[0,1],[1,0],[0,1]])
    M.append(pcm.FixedModel('A',A@A.T))
    M.append(pcm.FixedModel('B',B@B.T))
    M.append(pcm.FixedModel('I',np.eye(6)))
    MF=pcm.model.ModelFamily(M)

    # Now generate data from the full model with different values of theta
    trueModel = MF[-1]
    [cond_vec,part_vec]=pcm.sim.make_design(6,8)
    D = pcm.sim.make_dataset(trueModel,np.array([-1.0,-1.0,-1.0]),
                            signal=0.1,
                            n_sim = 20,
                            n_channel=20,part_vec=part_vec,
                            cond_vec=cond_vec)
    T,theta=pcm.fit_model_individ(D,MF,verbose=False)

    # pcm.vis.model_plot(T.likelihood-MF.num_comp_per_m)
    mposterior = MF.model_posterior(T.likelihood.mean(axis=0),method='AIC',format='DataFrame')
    cposterior = MF.component_posterior(T.likelihood,method='AIC',format='DataFrame')
    c_bf = MF.component_bayesfactor(T.likelihood,method='AIC',format='DataFrame')

    fig=plt.figure(figsize=(18,3.5))
    plt.subplot(1,3,1)
    pcm.vis.plot_tree(MF,mposterior.mean(axis=0),show_labels=True,show_edges=True)

    ax=plt.subplot(1,3,2)
    pcm.vis.plot_component(cposterior,type='posterior')

    ax=plt.subplot(1,3,3)
    pcm.vis.plot_component(c_bf,type='bf')

    pass

if __name__ == '__main__':
    sim_two_by_three()