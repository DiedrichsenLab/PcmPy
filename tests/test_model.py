"""
Unit test for model class
@author: jdiedrichsen
"""
import unittest 
import PcmPy as pcm 
import numpy as np 

class TestModel(unittest.TestCase): 
    
    def test_model_feature(self):
        """
        Test that it can sum a list of integers
        """
        A = np.zeros((4,3,10))
        M = pcm.ModelFeature("aModel",A)
        self.assertEqual(M.n_param,4)
        theta = np.array([0,1,2,3])
        G, dG = M.predict(theta)

    def test_model_component(self):
        C=np.zeros((3,5,5))
        C[0,0,0]=1
        C[1,2,2]=1
        C[2,4,4]=1
        theta = np.array([0,2,5])
        M = pcm.ModelComponent("bModel",C)
        self.assertEqual(M.n_param,3) 
        G, dG = M.predict(theta)
        self.assertEqual(G[2,2],np.exp(2))
        
    def test_model_free(self):
        A = np.eye(4)
        M = pcm.ModelFree('ceil',4)
        M.set_theta0(A)
        [G,dG] = M.predict(M.theta0)
        self.assertTrue(np.all((G-A)<1e-8))

if __name__ == '__main__':
    unittest.main()