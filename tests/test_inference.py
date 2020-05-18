#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_indicator

@author: jdiedrichsen
"""
# import sys
# sys.path.append('/Users/jdiedrichsen/Python')
import unittest
import PcmPy as pcm
import numpy as np
import pickle

f = open('/Users/jdiedrichsen/Python/PcmPy/demos/data_finger7T.p','rb')
Y,M = pickle.load(f)
f.close()

class TestInference(unittest.TestCase):

    def test_fixed_lik(self):
        YY = Y[0].measurements @ Y[0].measurements.T
        n_channel = Y[0].measurements.shape[1]
        Z = pcm.matrix.indicator(Y[0].obs_descriptors['cond_vec'])
        X = pcm.matrix.indicator(Y[0].obs_descriptors['part_vec'])
        theta = np.array([-0.5,0.5])
        lik,dL,d2L = pcm.inference.likelihood_individ(theta, M[0], YY, Z, X=X,
            n_channel=n_channel, return_deriv = 2,fit_scale=True)
        self.assertAlmostEqual(lik,46697.979838372456)
        self.assertAlmostEqual(dL[0],248.82177294654605)
        self.assertAlmostEqual(dL[1],13637.404410177838)
        self.assertAlmostEqual(d2L[0,0],409.189721036659)
        self.assertAlmostEqual(d2L[0,1],750.509995737473)
        self.assertAlmostEqual(d2L[1,1],29275.840725272497)

    def test_component_lik(self):
        MC=pcm.ModelComponent('muscle',M[0].G)
        YY = Y[0].measurements @ Y[0].measurements.T
        n_channel = Y[0].measurements.shape[1]
        Z = pcm.matrix.indicator(Y[0].obs_descriptors['cond_vec'])
        X = pcm.matrix.indicator(Y[0].obs_descriptors['part_vec'])
        theta = np.array([-0.5,0.5])
        lik,dL,d2L = pcm.inference.likelihood_individ(theta, MC, YY, Z, X=X,
            n_channel=n_channel, return_deriv = 2,fit_scale=False)
        self.assertAlmostEqual(lik,46697.979838372456)
        self.assertAlmostEqual(dL[0],248.82177294654605)
        self.assertAlmostEqual(dL[1],13637.404410177838)
        self.assertAlmostEqual(d2L[0,0],409.1897210366591)
        self.assertAlmostEqual(d2L[0,1],750.509995737473)
        self.assertAlmostEqual(d2L[1,1],29275.840725272497)

    def test_block_noise(self):
        MC=pcm.ModelComponent('muscle',M[0].G)
        YY = Y[0].measurements @ Y[0].measurements.T
        n_channel = Y[0].measurements.shape[1]
        Z = pcm.matrix.indicator(Y[0].obs_descriptors['cond_vec'])
        theta = np.array([-0.5,0.1,0.5])
        NM = pcm.model.BlockPlusIndepNoise(Y[0].obs_descriptors['part_vec'])
        lik,dL,d2L = pcm.inference.likelihood_individ(theta, MC, YY, Z, X=None,
            n_channel=n_channel, return_deriv = 2, fit_scale=False, Noise = NM)
        self.assertAlmostEqual(lik,56178.78716800276)
        self.assertAlmostEqual(dL[0], 669.9543080330623)
        self.assertAlmostEqual(dL[1],1568.8937559476763)
        self.assertAlmostEqual(dL[2],14100.867738004352)

    def test_fit_model_individ(self):
        MC = []
        MC.append(pcm.ModelComponent('muscle',M[0].G))
        MC.append(pcm.ModelComponent('natural',M[1].G))
        MC.append(pcm.ModelComponent('muscle+nat',[M[0].G,M[1].G]))
        theta0 = [np.ones((2,7)) * np.array([[-1,0.1]]).T]*2
        T, theta = pcm.inference.fit_model_individ(Y,MC,theta0=theta0)
        self.assertAlmostEqual(T.likelihood[0][1],-34923.790708900)

    def test_likelihood_group(self):
        MC = pcm.ModelComponent('muscle+nat',[M[0].G,M[1].G])
        MC.common_param=[True,False]
        n_subj = len(Y)
        Z = [None]*n_subj
        X = [None]*n_subj
        YY = [None]*n_subj
        n_channel = [None]*n_subj
        G_hat = [None]*n_subj
        Noise=[None]*n_subj

        for i in range(n_subj):
            Z[i], X[i], YY[i], n_channel[i], Noise[i], G_hat[i] = pcm.inference.set_up_fit(Y[i],run_effect = 'fixed')
        theta = np.array([1,0,-1,1,0,-1,1,0,-1,1,0,-1,1,0,-1,1,0,-1,1,0,-1,1])
        l,df,dff = pcm.inference.likelihood_group(theta, MC, YY, Z, X,
                       Noise=Noise,
                       n_channel=n_channel, fit_scale=True,
                       return_deriv=2)

    def test_fit_model_group(self):
        MC = []
        MC.append(pcm.ModelComponent('muscle',M[0].G))
        MC.append(pcm.ModelComponent('natural',M[1].G))
        MC.append(pcm.ModelComponent('muscle+nat',[M[0].G,M[1].G]))
        MC.append(pcm.ModelComponent('muscle+nat_2',[M[0].G,M[1].G]))
        MC[3].common_param=np.array([False,True])
        theta0 = [np.zeros((15,)),np.zeros((15,)),np.zeros((16,)),np.zeros(22,)]
        T, theta = pcm.inference.fit_model_group(Y, MC, fit_scale=True)
        self.assertAlmostEqual(T.likelihood[0][3]-T.likelihood[2][3],
                                 -160.533922713701) 

if __name__ == '__main__':
    unittest.main()