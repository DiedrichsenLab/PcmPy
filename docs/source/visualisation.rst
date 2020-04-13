Visualisation 
=============

Second Moment matrices
----------------------

One important way to visualize both the data and the model prediction is to show the second moment matrix as a colormap, for example using the command `imagesc`. The predicted second moment matrix for each mode is being returned by the fitting routines (see above) and can therefore directly visualized. A useful estimate for the empirical second moment matrix is the cross-validated estimate obtained using `pcm_estGCrossval`. Note that if you removed the block-effect using the runEffect option 'fixed' then you need to also remove it from the data to have a fair comparison. A simple way to do so is to center the estimate - i.e., to make the mean of the rows and columns of the second moment matrix zero. This is equivalent to subtracting out the mean pattern.  

.. sourcecode::python
   G_hat=pcm_estGCrossval(Y,partVec,condVec); 
   H = eye(5)-ones(5)/5;  % Centering matrix 
   imagesc(H*Gm*H'); 

Note also that you can transform a second moment matrix into a representational dissimilarity matrix (RDM) using the following equivalence (see Diedrichsen & Kriegeskorte, 2016): 

.. sourcecode::python
   C   = pcm_indicatorMatrix('allpairs',[1:numCond]'); 
   RDM = squareform(diag(C*G*C'));  

The RDM can also be plotted as the second-moment matrix. The only difference is that the RDM does not contain information about the baseline.  

Multidimensional scaling
------------------------

Another important way of visualizing the second moment matrix is Multi-dimensional scaling (MDS), an important technique in representational similarity analysis. When we look at the second moment of a population code, the natural way of performing this is classical multidimensional scaling. This technique plots the different conditions in a space defined by the first few eigenvectors of the second moment matrix - where each eigenvector is weighted by the $sqrt(\lambda)$. 

Importantly, MDS provides only one of the many possible 2- or 3-dimensional views of the high-dimensional representational structure. That means that one should never make inferences from this reduced view. It is recommended to look at as many different views of the representational structure as possible to obtain a unbiased impression. For high dimensional space, you surely will find *one* view that shows exactly what you want to show. There are a number of different statistical visualisation techniques that can be useful here, including the 'Grand tour' which provides a movie that randomly moves through different high-dimensional rotations. 

To enable the user to take a slightly more guided approach to visualizing, the function `pcm_classicalMDS` allows the specification of a contrast. The resultant projection is the subspace, in which the contrast is maximally represented. For example we can look at the finger patterns by looking at the difference between all the fingers: 


.. sourcecode::python
   C = pcm_indicatorMatrix('allpairs',[1:5]'); 
   COORD=pcm_classicalMDS(G,'contrast',C); 
   plot(COORD(:,1),COORD(:,2),'o'); 

The following command then provides a different projection, optimized for showing the difference between thumb and index and thumb and middle finger. The other fingers are then projected into this space 

.. sourcecode::python
   C=[1 -1 0 0 0;1 0 -1 0 0]'; 
   [COORD,l]=pcm_classicalMDS(Gm,'contrast',C);

Using different contrast makes especially sense when more conditions are studied, which can be described by different features or factors. These factors can then be used to provide a guide by which to rotate the representational structure. Note that when no contrast is provided, the functions attempts to represent the overall second moment matrix as good as possible. If the second moment is not centered, then the function chooses the  projection that best captures the activity of each condition relative to baseline (rest). This often results in a visualisation that is dominated by one mean vector (the average activity pattern).  

Plotting model evidence
-----------------------
Another approach to visualize model results is to plot the model evidence (i.e. marginal likelihoods). The marginal likelihoods are returned from the modeling routines in arbitrary units, and are thus better understood after normalizing to a null model at the very least. The lower normalization bound can be a null model, and upper bound is often a noise ceiling. This technique simply plots scaled likelihoods for each model fit.

.. sourcecode::python
   T = pcm_plotModelLikelihood(Tcross,M,'upperceil',Tgroup.likelihood(:,5)); 

Visualising the voxel-feature weights
-------------------------------------

Finally, like in an encoding model, we can also estimate and visualise the estimated voxel-feature weights or activity patterns, that are most likely under the current model. The PCM toolbox provides two ways of formulating an encoding model: First, the model features can be fixedly encoded in the design matrix **Z** and the data is modeled as:

.. math::
\mathbf{Y} = \mathbf{Z} \mathbf{U} + \boldsymbol{\epsilon}

In this case, **U** can be interpreted as the voxel-feature weights, and PCM estimates the variances and covariance of these feature maps, encoded in ${\mathbf{G}}(\boldsymbol{\theta})$.  You can get the BLUP estimates of **U** under model M  by calling

.. sourcecode::python

   U=pcm_estimateU(M,theta,Data,Z,X);

Secondly, in feature models, **Z** simply provides the basic asssignment of trials to conditions. The features themselves are encoded in the feature matrix **A**, which itself can flexibly be changed based on the parameters. 

..math::
\mathbf{Y} = \mathbf{ZA}(\boldsymbol{\theta})\mathbf{W}  + \boldsymbol{\epsilon}

For this case, you can obtain the best estimate of the voxel-feature weights and the currently best feature matrix by calling:

.. sourcecode::python

   [W,A]=pcm_estimateW(M,theta,Data,Z,X);`

Feature weights can then be visualized using standard techniques either in the volume or on the surface. Routines to do this are not included in the PCM toolbox.  