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
        data = pcm.dataset.Dataset(A)
        self.assertEqual(data.n_obs,10)
        self.assertEqual(data.n_channel,5)

    def test_make_design(self):
        cond_vec,part_vec = pcm.sim.make_design(4,8)
        self.assertEqual(cond_vec.size,32)

    def test_make_data(self):
        cond_vec,part_vec = pcm.sim.make_design(4,8)
        M = pcm.model.FixedModel("null",np.zeros((4,4)))
        D = pcm.sim.make_dataset(M,None,cond_vec,n_channel=40)
        self.assertEqual(D[0].measurements.shape[0],32)
        self.assertEqual(D[0].measurements.shape[1],40)

    def test_rand_seed(self):
        cond_vec,part_vec = pcm.sim.make_design(4,8)
        M = pcm.model.FixedModel("eye",np.eye(4))
        rng = np.random.default_rng(10)
        D1 = pcm.sim.make_dataset(M,None,cond_vec,signal=1,n_channel=20,rng=rng)
        rng = np.random.default_rng(10)
        D2 = pcm.sim.make_dataset(M,None,cond_vec,signal=1,n_channel=20,rng=rng)
        self.assertEqual(D1[0].measurements[0,1],D2[0].measurements[0,1])
        self.assertEqual(D1[0].measurements[10,11],D2[0].measurements[10,11])

if __name__ == '__main__':
    # a= TestSimulation()
    # a.test_rand_seed()
    unittest.main()