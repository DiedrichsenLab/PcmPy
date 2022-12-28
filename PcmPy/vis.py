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
import matplotlib.cm as cm
import matplotlib.colors as colors

import seaborn as sb
import pandas as pd



def model_plot(likelihood,null_model=0,noise_ceiling=None,upper_ceiling=None):
    """
    Make model comparision plot

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

def plot_tree(model_family,data,
            show_edges=True,
            show_labels=False,
            edge_color='diff_comp',
            edge_width=1,
            comp_colormap='tab20'):

    # Transform from pandas to numpy
    if type(data) in [pd.Series,pd.DataFrame]:
        data = data.to_numpy()

    # Set colormap for component-based coloring
    if isinstance(comp_colormap,str):
        comp_colormap = plt.get_cmap(comp_colormap) 
        comp_colormap = comp_colormap(np.arange(model_family.num_comp))

    # Make onedimensional and
    data = data.reshape((-1,))
    if data.shape[0]!=model_family.num_models:
        raise(NameError('data must have as many entries as model combinations'))

    # Get the layout from the model family
    [x,y]   = model_family.get_layout()

    # generate axis with appropriate labels
    ax = plt.gca()
    ax.set_aspect('equal')
    ax.set_xlim(np.min(x)-1,np.max(x)+1)
    ax.set_ylim(np.min(y)-1,np.max(y)+1)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Draw connections, if desired
    if show_edges:
        connect = model_family.get_connectivity(by_comp=True)
        any_connect = np.sum(connect,axis=0)
        [comp,fr,to]=np.where(connect==1)
        for i in range(fr.shape[0]):
            if edge_color == "uniform":
                l = mlines.Line2D([x[fr[i]], x[to[i]]],
                              [y[fr[i]], y[to[i]]],
                              color=(0,0,0,0.3),
                              zorder=1)
            elif edge_color == 'diff_comp':
                l = mlines.Line2D([x[fr[i]], x[to[i]]],
                              [y[fr[i]], y[to[i]]],
                              color=comp_colormap[comp[i]],
                              zorder=1)
            ax.add_line(l)

    # Determine color range and map
    cmap = cm.Reds
    norm = colors.Normalize(vmin=np.min(data), vmax=np.max(data))

    # Draw model circles
    for i in range(x.shape[0]):
        circle = plt.Circle((x[i], y[i]), 0.3,
                facecolor=cmap(norm(data[i])),
                edgecolor=(0,0,0,0.3),
                zorder = 30)
        ax.add_patch(circle)

    # Add labels to models
    if show_labels:
        for i in range(x.shape[0]):
            plt.text(x[i], y[i], model_family[i].name,
                zorder = 40)

def plot_component(data,type='posterior'):
    """Plots the result of a component analysis
    Args:
        data (_type_): _description_
        type (str, optional): _description_. Defaults to 'posterior'.
    """
    D = data.melt()
    ax = plt.gca()
    sb.barplot(data=D,x='variable',y='value')
    if (type=='posterior'):
        ax.set_ylabel('Posterior')
        plt.axhline(1/(1+np.exp(1)),color='k',ls=':')
        plt.axhline(0.5,color='k',ls='--')
    elif(type=='bf'):
        ax.set_ylabel('Bayes Factor')
        plt.axhline(0,color='k',ls='--')
    elif(type=='varestimate'):
        ax.set_ylabel('Variance Estimate')
    ax.set_xlabel('Component')