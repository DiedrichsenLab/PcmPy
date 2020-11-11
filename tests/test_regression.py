#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_indicator

@author: jdiedrichsen
"""
# import sys
# sys.path.append('/Users/jdiedrichsen/Python')
import sys
print(sys.path)
sys.path.append('/Users/jdiedrichsen/Python')
import unittest
import PcmPy as pcm
import numpy as np
import pickle
import matplotlib.pyplot as plt

# class TestInference(unittest.TestCase):
f = open('/Users/jdiedrichsen/Python/PcmPy/demos/data_regression_test.p','rb')
Z,Y,U,comp = pickle.load(f)
f.close()
Q = Y.shape[1]
theta = np.zeros((4,))

# Make model 
A = pcm.regression.RidgeDiag(comp, fit_intercept = False)


class TestRegression(unittest.TestCase):

    def test_likelihoods(self): 
        l1,d1,D1 = pcm.regression.likelihood_diagYYT(theta, Z, Y @ Y.T , Q, comp, return_deriv=2)
        l2, d2, D2 = pcm.regression.likelihood_diagYTY(theta, Z, Y, comp, return_deriv=2)
        self.assertAlmostEqual(l1,l2)
        self.assertAlmostEqual(d1[0],d2[0])

    def test_constructor(self):
        A = pcm.regression.RidgeDiag(comp, fit_intercept = False)

    def test_optimize(self): 
        A = pcm.regression.RidgeDiag(comp, fit_intercept = False)
        A.optimize_regularization(Z,Y)
        A.fit(Z,Y)
        Yp = A.predict(Z)

if __name__ == '__main__':
    unittest.main()