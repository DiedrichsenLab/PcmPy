import numpy as np
import PcmPy as pcm
from PcmPy import sim
import pandas as pd
import plotly.io as pio
import plotly.graph_objects as go
import scipy.stats as ss

margdict = dict(l=10,r=10, b=10, t=10, pad=4)


def get_corr(X,cond_vec):
    """
        Get normal correlation
    """
    p1 = X[cond_vec==0,:].mean(axis=0)
    p2 = X[cond_vec==1,:].mean(axis=0)
    return np.corrcoef(p1,p2)[0,1]

def get_noiseceil(X,cond_vec):
    """
        Calculate noise ceiling over reliabilities
    """
    rel = np.array([0.0,0.0])
    for i in [0,1]:
        N = np.sum(cond_vec==i) # Number of measurements
        R = np.corrcoef(X[cond_vec==i,:]) # Correlation matrix
        index_R = np.where(~np.eye(N,dtype=bool)) # Average cross-block correlations
        r = np.mean(R[index_R]) # Average the non-diagnal elements.
        rel[i] = r * N / (r*(N-1)+1) # Overall realibility of the mean

    # Check if both are above zero
    if rel[0]>0 and rel[1]>0:
        noise_r = np.sqrt(rel[0]*rel[1])
    else:
        noise_r = np.nan
    return noise_r


def get_crossblock(X,cond_vec,part_vec):
    """
        calculate the cross-block correlation
    """
    G = pcm.util.est_G_crossval(X,cond_vec,part_vec)
    var = G[0][0,0]*G[0][1,1]
    if var<=0:
        crosscorr = np.nan
    else:
        crosscorr = G[0][0,1]/np.sqrt(var)
    return crosscorr


def do_sim(corr,signal=np.linspace(0,5,20),n_sim=50,randseed=None):
    M = pcm.CorrelationModel('corr',num_items=1,corr=corr,cond_effect=False)
    G,dG = M.predict([0,0])
    cond_vec,part_vec = pcm.sim.make_design(2,2)
    Lcorr = []
    LnoiseCeil = []
    Lsign = []
    LcrossBlock =[]
    rng = np.random.default_rng(randseed)

    for s in signal:
        D = pcm.sim.make_dataset(M, [0,0], cond_vec, n_sim=n_sim, signal=s,rng=rng)
        for i in range(n_sim):
            data = D[i].measurements
            Lcorr.append(get_corr(data,cond_vec))
            LnoiseCeil.append(get_noiseceil(data,cond_vec))
            LcrossBlock.append(get_crossblock(data,cond_vec,part_vec))
            Lsign.append(s)
    S = pd.DataFrame({'r_naive':Lcorr,'signal':Lsign,
                        'noiseCeil':LnoiseCeil,'cross_block':LcrossBlock})
    S['true'] = np.ones((S.shape[0],))*corr
    return S

def do_sim_corrpcm(corr=0.7,signal=0.5,n_sim=20,nsteps = 11,randseed=None):
    # Make the design in this case it's 2 runs, 2 conditions!
    cond_vec,part_vec = pcm.sim.make_design(2,2)
    # Generate different models from 0 to 1
    M=[]
    for r in np.linspace(0,1,nsteps):
        M.append(pcm.CorrelationModel(f"{r:0.2f}",num_items=1,corr=r,cond_effect=False))
    Mflex = pcm.CorrelationModel("flex",num_items=1,corr=None,cond_effect=False)
    M.append(Mflex)
    # Now do the simulations and fit with the models
    rng = np.random.default_rng(randseed)
    Mtrue = pcm.CorrelationModel('corr',num_items=1,corr=corr,cond_effect=False)
    D = pcm.sim.make_dataset(Mtrue, [0,0], cond_vec,part_vec=part_vec,n_sim=n_sim, signal=signal, rng=rng)
    T,theta = pcm.inference.fit_model_individ(D,M,fixed_effect=None,fit_scale=False)
    return T,theta,M

def plot_Figure2(D,T,Tstd):
    # This code generates an interactive Figure for Figure2, using plotly
    fig = go.Figure()

    marker=dict(color='rgba(0, 200, 0, 0.04)', size=10)
    fig.add_scatter(x=D.signal,y=D.r_naive,
                name='individual simulations',mode='markers',
                marker=marker,hoverinfo='skip',
                showlegend=False)

    # Make the hover-template: Once that has been set,
    # hoverinfo does not have an effect any more
    hoverT = '<i>Signal</i>: %{x:.2f}<br>Mean: %{y:.2f}<br>%{text}'
    # Make the text for each point
    text = []
    for s in Tstd['r_naive']:
        text.append(f"Std: {s:.2f}")

    fig.add_scatter(x=T.signal, y=T.r_naive,
                name='',
                text = text,
                hovertemplate = hoverT,
                line = dict(color='rgba(0, 100, 0, 1)', width=4),
                mode = 'lines',
                showlegend=False)

    hoverT = 'True correlation<br>%{y:.2f}'

    fig.add_scatter(x=T.signal, y=T.true,
                name='',
                hovertemplate = hoverT,
                line = dict(color='rgba(0, 0, 0, 1)', width=1, dash='dash'),
                mode = 'lines',
                showlegend=False)


    fig.update_layout(
            hovermode='closest',
            autosize=True, # width =xx, heigh =xxx
            template = 'plotly_white',
            yaxis=dict(
                title_text="Correlation",
                titlefont=dict(size=18)
            ),
            xaxis=dict(
                title_text="Signal to Noise",
                titlefont=dict(size=18)
            ),
            margin = margdict
        )
    return(fig)

def plot_Figure3(D,T,Tstd):
    # This code generates an interactive Figure for Figure2, using plotly
    fig = go.Figure()

    lines = ['r_naive','corr_corrected','corr_corrected_imp','cross_block','cross_block_imp','true']
    linestyle = [dict(color='rgba(0, 100, 0, 1)', width=3),
                 dict(color='rgba(0, 0, 150, 1)', width=3),
                 dict(color='rgba(0, 0, 150, 1)', width=3,dash='dash'),
                 dict(color='rgba(150, 0, 0, 1)', width=3),
                 dict(color='rgba(150, 0, 0, 1)', width=3,dash='dash'),
                 dict(color='rgba(0, 0, 0, 1)', width=1, dash='dash')]
    shadecolor = ['rgba(0, 100, 0, 0.1)',
                   'rgba(0, 0, 150, 0.1)',
                   'rgba(0, 0, 150, 0.1)',
                   'rgba(150, 0, 0, 0.1)',
                   'rgba(150, 0, 0, 0.1)',
                   'rgba(0, 0, 0, 0.1)']
    hoverT = ['Naive Correlation<br>Mean: %{y:.2f}<br>%{text}',
            'Noise-ceiling (exclusion): <br>Mean: %{y:.2f}<br>%{text}',
            'Noise-ceiling (imputation):<br>Mean: %{y:.2f}<br>%{text}',
            'Crossblock (exclusion):<br>Mean: %{y:.2f}<br>%{text}',
            'Crossblock (imputation):<br>Mean: %{y:.2f}<br>%{text}',
            'True correlation<br>%{y:.2f}']

    for i,line in enumerate(lines):
        # Make the text for each point
        text = []
        for s in Tstd[line]:
            text.append(f"Std: {s:.2f}")

        x = T.signal.to_numpy()
        y = T[line].to_numpy()
        y_upper = y + Tstd[line].to_numpy()/np.sqrt(50)
        y_lower = y - Tstd[line].to_numpy()/np.sqrt(50)

        fig.add_scatter(x=T.signal, y=T[line],
                name='',
                text = text,
                hovertemplate = hoverT[i],
                line = linestyle[i],
                mode = 'lines',
                showlegend=False)
        if line !='true':
            fig.add_scatter(x=np.concatenate([x,x[::-1]]), # x, then x reversed
            y=np.concatenate([y_upper,y_lower[::-1]]), # upper, then lower reversed
            fill='toself',
            fillcolor=shadecolor[i],
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            showlegend=False)

    fig.update_layout(
            hovermode='x',
            autosize=True, # width =xx, heigh =xxx
            template = 'plotly_white',
            yaxis=dict(
                title_text="Correlation",
                titlefont=dict(size=18)
            ),
            xaxis=dict(
                title_text="Signal to Noise",
                titlefont=dict(size=18),
                range=[0,3.1]),
            margin = margdict
        )
    return(fig)

def plot_Figure4(T,theta,M):
    # This code generates an interactive Figure for Figure4, using plotly
    fig = go.Figure()

    lines = []

    markerstyle = [dict(color='rgba(0, 50, 50, 0.3)', size=7),
                 dict(color='rgba(150, 0, 0, 1)', size=7)]
    linestyle = [dict(color='rgba(0, 50, 50, 0.2)', width=2),
                 dict(color='rgba(150, 0, 0, 1)', width=2)]
    shadecolor = ['rgba(150, 0, 0, 0.1)']
    hoverT = ['Maximum Likelihood%{x:.2f}',
            'Mean Likelihood%{y:.2f}']

    flexL = T.likelihood.iloc[:,-1].to_numpy()
    flex_r = M[-1].get_correlation(theta[-1])
    fixL = T.likelihood.iloc[:,0:-1].to_numpy()
    n= fixL.shape[1]
    fix_r = np.empty((n,))
    for i in range(n):
        fix_r[i]= M[i].corr
    # recenter the correlation
    mfixL = fixL.mean(axis=1)
    fixL = fixL - mfixL.reshape(-1,1)
    flexL = flexL - mfixL  # apply same scaling
    # Plot individual linees
    for i in range(fixL.shape[0]):
        fig.add_scatter(x=fix_r, y=fixL[i,:],
            name='',
            line = linestyle[0],
            mode = 'lines',
            hoverinfo="skip",
            showlegend=False)

    # Plot the maximum
    fig.add_scatter(x=flex_r, y=flexL,
            name='',
            marker = markerstyle[0],
            mode = 'markers',
            hovertemplate=hoverT[0],
            showlegend=False)

    # Now plot the mean line with
    x = fix_r
    y = fixL.mean(axis=0)
    fig.add_scatter(x=fix_r, y=y,
        name='',
        line = linestyle[1],
        hovertemplate=hoverT[1],
        mode = 'lines',
        showlegend=False)

    # Do the maximal value of the mean
    maxy=np.max(y)
    maxr=x[np.argmax(y)]
    fig.add_scatter(x=np.array([maxr]), y=np.array([maxy]),
        name='',
        marker = markerstyle[1],
        hovertemplate='Max %{x:.2f}\n%{y:.2f}',
        mode = 'markers',
        showlegend=False)

    s = fixL.std(axis=0)/np.sqrt(fixL.shape[0])
    y_upper = y + s
    y_lower = y - s
    fig.add_scatter(x=np.concatenate([x,x[::-1]]), # x, then x reversed
    y=np.concatenate([y_upper,y_lower[::-1]]), # upper, then lower reversed
    fill='toself',
    fillcolor=shadecolor[0],
    line=dict(color='rgba(255,255,255,0)'),
    hoverinfo="skip",
    showlegend=False)

    fig.update_layout(
            hovermode='closest',
            hoverdistance = 10,
            autosize=True, # width =xx, heigh =xxx
            template = 'plotly_white',
            yaxis=dict(
                title_text="Difference in Log-Likelihood",
                titlefont=dict(size=18)
            ),
            xaxis=dict(
                title_text="Correlation",
                titlefont=dict(size=18),
                range=[0,1.02]),
            margin=margdict)
    return(fig)


def dosim_2():
    # Make a spacing of different signals
    sig = np.linspace(0.1,5.1,21)# np.logspace(np.log(0.2),np.log(5),10)
    # Get the simulations
    D = do_sim(0.7,n_sim=200,signal=sig,randseed=10)
    # Summarize the mean and std deviation
    T = D.groupby("signal").apply(np.mean)
    Tstd = D.groupby("signal").apply(np.std)
    # Plot and show the Figure
    fig = plot_Figure2(D,T,Tstd)
    fig.write_html("Figure_2.html",include_plotlyjs='cdn',full_html=False)
    fig.show()

def dosim_3():
    sig = np.array([0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.5,0.6,0.7,0.8,0.9,1.5,2,2.5,3])# np.logspace(np.log(0.2),np.log(5),10)
    D = do_sim(0.7,n_sim=10000,signal=sig,randseed=12)
    D['noiseceil_nan'] = np.isnan(D.noiseCeil)
    D['corr_corrected'] = D.r_naive / D.noiseCeil
    D['corr_corrected_imp'] = D.corr_corrected
    D.loc[np.isnan(D.corr_corrected),'corr_corrected_imp']=0
    D['cross_block_imp'] = D.cross_block
    D.loc[np.isnan(D.cross_block),'cross_block_imp']=0
    T = D.groupby("signal").apply(np.mean)
    Tstd = D.groupby("signal").apply(np.std)
    fig = plot_Figure3(D,T,Tstd)
    fig.write_html("Figure_3.html",include_plotlyjs='cdn',full_html=False)
    fig.show()

def dosim_4():
    T,theta,M = do_sim_corrpcm(corr=0.7,signal=0.5,n_sim=20,nsteps=21,randseed=11)
    fig=plot_Figure4(T,theta,M)
    fig.write_html("Figure_4.html",include_plotlyjs='cdn',full_html=False)
    fig.show()
    fixL = T.likelihood.iloc[:,0:-1].to_numpy()
    # recenter the correlation
    mfixL = fixL.mean(axis=1)
    fixL = fixL - mfixL.reshape(-1,1)
    n= fixL.shape[1]
    fix_r = np.empty((n,))
    for i in range(n):
        fix_r[i]= M[i].corr
    R1=ss.ttest_rel(fixL[:,14],fixL[:,20])
    print(R1)

    pass




if __name__ == "__main__":
    dosim_4()
    pass