#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_indicator

@author: jdiedrichsen
"""
# Import necessary libraries
import pystan as ps
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb

def school_model():
    schools_code = """
    data {
        int<lower=0> J;         // number of schools
        array[J] real y;              // estimated treatment effects
        array[J] real<lower=0> sigma; // standard error of effect estimates
    }
        parameters {
        real mu;                // population treatment effect
        real<lower=0> tau;      // standard deviation in treatment effects
        vector[J] eta;          // unscaled deviation from mu by school
    }
        transformed parameters {
        vector[J] theta = mu + tau * eta;        // school treatment effects
    }
    model {
        target += normal_lpdf(eta | 0, 1);       // prior log-density
        target += normal_lpdf(y | theta, sigma); // log-likelihood
    }
    """
    schools_data = {"J": 8,
                "y": [28,  8, -3,  7, -1,  1, 18, 12],
                "sigma": [15, 10, 16, 11,  9, 11, 10, 18]}
    sm = ps.StanModel(model_code=schools_code)
    fit = sm.sampling(data=schools_data, iter=1000, chains=4, seed=1)
    eta = fit["eta"] 
    df = fit.to_frame()

if __name__ == '__main__':
    df=school_model()
    pass