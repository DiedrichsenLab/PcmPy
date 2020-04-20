#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_indicator 

@author: jdiedrichsen
"""

import unittest 
import PcmPy as pcm 
import numpy as np 

class TestIndicator(unittest.TestCase): 
    
    def test_identity(self):
        a = np.array(range(0,5))
        a = np.concatenate((a,a))
        X = pcm.matrix.indicator(a)
        n_row,n_col = X.shape
        self.assertEqual(n_row,10)
        self.assertEqual(n_col,5)

    def test_identity_pos(self):
        a = np.array(range(0,5))
        a = np.concatenate((a,a))
        X = pcm.matrix.indicator(a,positive=True)
        n_row,n_col = X.shape
        self.assertEqual(n_row,10)
        self.assertEqual(n_col,4)

    def test_pairwise(self):
        a = np.array(range(0,5))
        X = pcm.matrix.pairwise_contrast(a)
        n_row,n_col = X.shape
        self.assertEqual(n_row,10)
        self.assertEqual(n_col,5)

if __name__ == '__main__':
    unittest.main()        