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

# Make model 
A = pcm.regression.RidgeDiag(comp, fit_intercept = False)


class TestRegression(unittest.TestCase):

    def test_constructor(self):
        A = pcm.regression.RidgeDiag(comp, fit_intercept = False)
        self.assert(A.n_param,4)

    def test_optimize(self): 
        A = pcm.regression.RidgeDiag(comp, fit_intercept = False)
        A.optimize_regularization(Z,Y)
        A.fit(Z,Y)
        Yp = A.predict(Z)
        self.assertAlmostEqual(dL[0],248.82177294654605)

if __name__ == '__main__':
    unittest.main()