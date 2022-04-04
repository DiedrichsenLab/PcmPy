#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions for visualization of PCM models, Data, and model fits
@author: jdiedrichsen
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as mlines
import seaborn as sb
import pandas as pd


def model_plot(likelihood,null_model=0,noise_ceiling=None,upper_ceiling=None):
    """
    Make model comparisiom plot

    Parameters:
        likelihood (pd.DataFrame)
            Data Frame with the results (from T.likelihood)
        null_model (int or string)
            Number or name of the model that define the zero-point
        noise_ceiling(int or string)
            Number or name of the model that defines the noise ceiling
        upper_ceiling (np.array or series)
            Likelihood for the upper noise ceiling (usuallu from group fit)
    Returns:
        ax (matplotlib.Axis.axis)
            Matplotlib axis object

    """

    noise_ceil_col = [0.5, 0.5, 0.5, 0.2]

    m_names = likelihood.columns.values
    if type(null_model) != str:
        null_model = m_names[null_model]
    if noise_ceiling is not None:
        if type(noise_ceiling) != str:
            noise_ceiling = m_names[noise_ceiling]

    # Subtract the baseline
    baseline = likelihood.loc[:,null_model].values
    likelihood = likelihood - baseline.reshape(-1,1)

    # Stretch out the data frame
    LL=pd.melt(likelihood)
    indx = np.logical_and(LL.model !=null_model, LL.model !=noise_ceiling)
    ax = sb.barplot(x=LL.model[indx], y=LL.value[indx])
    xlim = ax.get_xlim()
    if noise_ceiling is not None:
        noise_lower = np.nanmean(likelihood[noise_ceiling])
        if upper_ceiling is not None:
            noise_upper = np.nanmean(upper_ceiling-baseline)
            noiserect = patches.Rectangle((xlim[0], noise_lower), xlim[1]-xlim[0], noise_upper-noise_lower, linewidth=0, facecolor=noise_ceil_col, zorder=1e6)
            ax.add_patch(noiserect)
        else:
            l = mlines.Line2D([xlim[0], xlim[1]], [noise_lower, noise_lower],color=[0,0,0], linestyle=':')
            ax.add_line(l)
    ax.set_ylabel('Log Bayes Factor')
    return ax