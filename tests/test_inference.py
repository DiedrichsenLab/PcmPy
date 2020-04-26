#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_indicator 

@author: jdiedrichsen
"""
import sys
print(sys.path)
sys.path.append('/Users/jdiedrichsen/Python')
import unittest
import PcmPy as pcm 
import numpy as np 
import pickle

# class TestInference(unittest.TestCase):
Y,M = pickle.load(open('/Users/jdiedrichsen/Python/PcmPy/demos/data_finger7T.p','rb'))
YY = Y[0].measurements @ Y[0].measurements.T
n_channel = Y[0].measurements.shape[1]
Z = pcm.matrix.indicator(Y[0].obs_descriptors['cond_vec'])
X = pcm.matrix.indicator(Y[0].obs_descriptors['part_vec'])
theta = np.array([0])
lik = pcm.inference.likelihood_individ(theta, M[0], YY, Z, X=X,
                                        n_channel=n_channel)
# lik 42295.4215095062
# if __name__ == '__main__':
#     unittest.main()        