#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_indicator 

@author: jdiedrichsen
"""

import unittest 
import PcmPy as pcm 
import numpy as np 

class TestSimulation(unittest.TestCase): 

    def test_dataset(self):
        A = np.zeros((10,5))
        data = pcm.Dataset(A)
        self.assertEqual(data.n_obs,10)
        self.assertEqual(data.n_channel,5)

    def test_make_design(self):
        cond_vec,part_vec = pcm.sim.make_design(4,8)
        self.assertEqual(cond_vec.size,32)

    def test_make_data(self):
        cond_vec,part_vec = pcm.sim.make_design(4,8)
        M = pcm.FixedModel("null",np.zeros((4,4)))
        D = pcm.sim.make_dataset(M,None,cond_vec,n_channel=40)
        self.assertEqual(D[0].measurements.shape[0],32)
        self.assertEqual(D[0].measurements.shape[1],40)

if __name__ == '__main__':
    unittest.main()        