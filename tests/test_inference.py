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

if __name__ == '__main__':
    unittest.main()