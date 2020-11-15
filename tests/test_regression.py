#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_indicator

@author: jdiedrichsen
"""
import sys
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
P = Y.shape[1]
theta = np.zeros((4,))

# Make model
A = pcm.regression.RidgeDiag(comp, fit_intercept = False)


class TestRegression(unittest.TestCase):

    def test_likelihood(self):
        comp  = np.array([0,0,0,1,1,1,2,2]) # 20 regressors in 4 groups
        Q = comp.shape[0]
        theta = np.array([-1,-2,0.5,1.2]) # Theta's for simulation
        P = 10
        N = 500
        Y = np.random.normal(0,1,(N,P))
        YY = Y @ Y.T
        Z = np.random.normal(0,1,(N,Q))
        X = None
        l1,dl1,ddl1 = pcm.regression.likelihood_diagYYT_ZZT(theta, Z , YY, P , comp, X,return_deriv=2)
        l2,dl2,ddl2 = pcm.regression.likelihood_diagYYT_ZTZ(theta, Z , YY, P , comp, X,return_deriv=2)
        l3,dl3,ddl3 = pcm.regression.likelihood_diagYTY_ZZT(theta, Z , Y, comp, X,return_deriv=2)
        l4,dl4,ddl4 = pcm.regression.likelihood_diagYTY_ZTZ(theta, Z , Y, comp, X,return_deriv=2)
        self.assertAlmostEqual(l1,l2)
        self.assertAlmostEqual(l1,l3)
        self.assertAlmostEqual(l1,l4)
        for i in range(4):
            self.assertAlmostEqual(dl1[i],dl2[i])
            self.assertAlmostEqual(dl1[i],dl3[i])
            self.assertAlmostEqual(dl1[i],dl4[i])
        for i in range(4):
            for j in range(4):
                self.assertAlmostEqual(ddl1[i,j],ddl2[i,j])
                self.assertAlmostEqual(ddl1[i,j],ddl3[i,j])
                self.assertAlmostEqual(ddl1[i,j],ddl4[i,j])

    def test_constructor(self):
        A = pcm.regression.RidgeDiag(comp, fit_intercept = False)

    def test_optimize(self):
        A = pcm.regression.RidgeDiag(comp, fit_intercept = False)
        A.optimize_regularization(Z,Y)
        A.fit(Z,Y)
        Yp = A.predict(Z)

if __name__ == '__main__':
    unittest.main()