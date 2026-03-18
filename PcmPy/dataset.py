#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Definition of PCM Dataset class and subclasses
@author: baihan, jdiedrichsen
"""
import numpy as np
import copy

class Dataset:
    """Class for holding a single data set with observation descriptors.

    Defines members that every class needs to have, but does not
    implement any interesting behavior. Defines a light version of the RSA
    Dataset class (which could be used instead)

    Parameters:
        measurements (np.ndarray):
            n_obs x n_channel 2d-array,
        descriptors (dict):
            descriptors (metadata)
        obs_descriptors (dict):
            observation descriptors (all are array-like with shape = (n_obs,...))
        channel_descriptors (dict):
            channel descriptors (all are array-like with shape = (n_channel,...))
    """

    def __init__(self, measurements, descriptors=None,
                 obs_descriptors=None, channel_descriptors=None):
        if measurements.ndim != 2:
            raise AttributeError(
                "measurements must be in dimension n_obs x n_channel")
        self.measurements = measurements
        self.n_obs, self.n_channel = self.measurements.shape
        self.descriptors = descriptors
        self.obs_descriptors =obs_descriptors
        self.channel_descriptors = channel_descriptors

def combine_datasets(D,indx=None):
    """ Combines datasets for a given index
    Args:
        D (list): List of PCM datasets
        indx (ndarray): Index of datasets to combine (default 1..N)
    Returns:
        Dc: Combined dataset
    """
    if indx is None:
        indx = np.arange(len(D))
    Dc = copy.deepcopy(D[indx[0]])
    for i in indx[1:]:
        Dc.measurements=np.append(Dc.measurements,D[i].measurements,axis=1)
    return Dc
