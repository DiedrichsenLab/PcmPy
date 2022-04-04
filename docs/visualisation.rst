Visualisation
=============

Second Moment matrices
----------------------

One important way to visualize both the data and the model prediction is to plot the second moment matrix as a colormap, for example using the matplotlib command ``plt.imshow``. The predicted second moment matrix for a fitted model can be obtained using ``my_model.predict(theta)``. For the data we can get a cross-validated estimate obtained using the function :meth:`util.est_G_crossval`. Note that if you removed the block-effect using the runEffect option 'fixed' then you need to also remove it from the data to have a fair comparison.

Note also that you can transform a second moment matrix into a representational dissimilarity matrix (RDM) using the following equivalence (see Diedrichsen & Kriegeskorte, 2016):

.. sourcecode::python
   from scipy.spatial.distance import squareform
   C = pcm.matrix.pairwise_contrast(np.arange(5))
   RDM = squareform(np.diag(C @ G_hat[0,:,:]@C.T))
   plt.imshow(RDM)

The only difference is that the RDM does not contain information about the baseline.

Multidimensional scaling
------------------------

Another important way of visualizing the second moment matrix is Multi-dimensional scaling (MDS), an important technique in representational similarity analysis. When we look at the second moment of a population code, the natural way of performing this is classical multidimensional scaling. This technique plots the different conditions in a space defined by the first few eigenvectors of the second moment matrix - where each eigenvector is weighted by the $sqrt(\lambda)$.

Importantly, MDS provides only one of the many possible 2- or 3-dimensional views of the high-dimensional representational structure. That means that one should never make inferences from this reduced view. It is recommended to look at as many different views of the representational structure as possible to obtain a unbiased impression. For high dimensional space, you surely will find *one* view that shows exactly what you want to show. There are a number of different statistical visualisation techniques that can be useful here, including the 'Grand tour' which provides a movie that randomly moves through different high-dimensional rotations.

Classical multidimensional scaling from the matlab version still needs to be implemented in Python.

Plotting model evidence
-----------------------
Another approach to visualize model results is to plot the model evidence (i.e. marginal likelihoods). The marginal likelihoods are returned from the modeling routines in arbitrary units, and are thus better understood after normalizing to a null model at the very least. The lower normalization bound can be a null model, and upper bound is often a noise ceiling. This technique simply plots scaled likelihoods for each model fit.

.. sourcecode::python
   pcm.vis.model_plot(T_in.likelihood,
                        null_model = 'null',
                        noise_ceiling= 'ceil')

See :ref:`examples` for a practical example for this.
