#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_indicator

@author: jdiedrichsen
"""
# Import necessary libraries
import PcmPy as pcm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from numpy import exp, sqrt

if __name__ == '__main__':
    M = pcm.CorrelationModel('corr',num_items=3,corr=None,cond_effect=True)
    G,dG = M.predict([0,-0.5,-2,-1.0,0.9])
    pass
