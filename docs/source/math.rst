Mathematical and algorithmical details
======================================

Likelihood
----------

In this section, we derive the likelihood in the case that there are no fixed effects. In this case the distribution of the data would be

.. math::
    \begin{array}{c}
    {\bf{y}} \sim N \left(0,{\bf{V}} \right)\\ {\bf{V}}=\bf{ZGZ^{T}+S}\sigma^{2}_{\epsilon}
    \end{array}

To calculate the likelihood, let us consider at the level of the single voxel, namely, :math:`\mathbf{Y}=[\mathbf{y_1},\mathbf{y_2},...,\mathbf{y_p}]`. Then the likelihood over all voxels, assuming that the voxels are independent (e.g. effectively pre-whitened) is

.. math::
    p \left( {\bf{Y}}|{\bf{V}} \right)= \prod^{P}_{i=1} (2\pi)^{-\frac{N}{2}} |{\bf{V}}|^{-\frac{1}{2}} exp \left( -\frac{1}{2}{\bf{y}}_i^T {\bf{V}}^{-1} {\bf{y}}_i \right)

When we take the logarithm of this expression, the product over the individual Gaussian probabilities becomes a sum and the exponential disappears:

.. math::
    L=\mathrm{ln}\left(p\left(\bf{Y}|V\right)\right) = \sum_{i=1}^{P} \mathrm{ln}  p\left(\bf{y}_{i}\right)\\

.. math::
    =-\frac{NP}{2}\mathrm{ln}\left(2\pi \right)-\frac{P}{2}\mathrm{ln}\left(|\bf{V}|\right)-\frac{1}{2}\sum _{i=1}^{P}{\bf{y}}_{i}^{T}{\bf{V}}^{-1}{\bf{y}}_{i}

.. math::
    =-\frac{NP}{2}\mathrm{ln} \left(2\pi \right)
    -\frac{P}{2}\mathrm{ln}\left(|\bf{V}|\right)
    -\frac{1}{2} trace \left({\bf{Y}}^{T}{\bf{V}}^{-1} \bf{Y} \right)


Using the trace trick, which allows :math:`\mathrm{trace}\left(\bf{ABC}\right) = \mathrm{trace}\left(\bf{BCA}\right)`, we can obtain a form of the likelihood that does only depend on the second moment of the data, :math:`\bf{YY}^{T}` ,as a sufficient statistics:

.. math::
    L =-\frac{NP}{2}\mathrm{ln}\left(2\pi \right)-\frac{P}{2}\mathrm{ln}\left(|\bf{V}|\right)-\frac{1}{2}trace\left({\bf{Y}\bf{Y}}^{T}{\bf{V}}^{-1}\right)

Restricted likelihood
---------------------

In the presence of fixed effects (usually effects of no interest), we have the problem that the estimation of these fixed effects depends iterativly on the current estimate of :math:`\bf{V}` and hence on the estimates of the second moment matrix and the noise covariance.

.. math::
    {\bf{\hat{B}}} =
    \left( {\bf{X}}^T {\bf{V}}^{-1} {\bf{X}} \right)^{-1}
    {\bf{X}}^T{\bf{V}}^{-1}{\bf{Y}}

Under the assumption of fixed effects, the distribution of the data is

.. math::
    {\bf{y_i}} \sim N \left(\bf{Xb_i},{\bf{V}} \right)

To compute the likelihood we need to remove these fixed effects from the data, using the residual forming matrix

.. math::
    {\bf{R}} = \bf{I} - \bf{X}{\left( {{{\bf{X}}^T}{{\bf{V}}^{ - 1}}{\bf{X}}} \right)^{ - 1}}{{\bf{X}}^T}{{\bf{V}}^{ - 1}}

.. math::
    {\bf{r_i}} = \bf{Ry_i}

For the optimization of the random effects we therefore also need to take into account the uncertainty in the fixed effects estimates. Together this leads to a modified likelihood - the restricted likelihood.

.. math::
    L_{ReML} =-\frac{NP}{2}\mathrm{ln}\left(2\pi \right)-\frac{P}{2}\mathrm{ln}\left(|\bf{V}|\right)-\frac{1}{2}trace\left({\bf{Y}\bf{Y}}^{T}{\bf{R}}^{T}{\bf{V}}^{-1}\bf{R}\right)-\frac{P}{2}\mathrm{ln}|\bf{X}^{T}\bf{V}^{-1}\bf{X}|

Note that the third term can be simplified by noting that

.. math::
    \bf{R}^{T}{\bf{V}}^{-1}\bf{R} = \bf{V}^{-1} - \bf{V}^{-1}\bf{X} (\bf{X}{\bf{V}}^{-1}\bf{X})^{-1}\bf{X}^{T}\bf{V}^{-1}=\bf{V}^{-1}\bf{R}=\bf{V}_{R}^{-1}

First derivatives of the log-likelihood
---------------------------------------
Next, we find the derivatives of *L* with respect to each hyper parameter :math:`\theta_{i}`, which influence G. Also we need to estimate the hyper-parameters that describe the noise, at least the noise parameter :math:`\sigma_{\epsilon}^{2}`. To take these derivatives we need to use two general rules of taking derivatives of matrices (or determinants) of matrices:

.. math::
    \frac{{\partial \ln \left|{\bf{V}} \right|}}{{\partial {\theta _i}}} = trace\left( {{{\bf{V}}^{ - 1}}\frac{{\partial {\bf{V}}}}{{\partial {\theta _i}}}} \right)

.. math::
    \frac{{\partial {{\bf{V}}^{ - 1}}}}{{\partial {\theta _i}}} = {{\bf{V}}^{ - 1}}\left( {\frac{{\partial {\bf{V}}}}{{\partial {\theta _i}}}} \right){{\bf{V}}^{ - 1}}


Therefore the derivative of the log-likelihood in [@eq:logLikelihood]. in respect to each parameter is given by:

.. math::
    \frac{{\partial {L_{ML}}}}{{\partial {\theta _i}}} = - \frac{P}{2}trace\left( {{{\bf{V}}^{ - 1}}\frac{{\partial {\bf{V}}}}{{\partial {\theta _i}}}} \right) + \frac{1}{2}trace\left( {{{\bf{V}}^{ - 1}}\frac{{\partial {\bf{V}}}}{{\partial {\theta _i}}}{{\bf{V}}^{ - 1}}{\bf{Y}}{{\bf{Y}}^T}} \right)

First derivatives of the restricted log-likelihood
--------------------------------------------------

First, let’s tackle the last term of the restricted likelihood function:

.. math::
    l = -\frac{P}{2}\ln|\mathbf{X}^T\mathbf{V}^{-1}\mathbf{X}|

.. math::
    \frac{\partial{l}}{\partial{\theta_i}} = -\frac{P}{2}trace\left( \left(\mathbf{X}^T\mathbf{V}^{-1}\mathbf{X} \right)^{-1}\mathbf{X}^T\frac{\partial{\mathbf{V}^{-1}}}{\partial{\theta_i}}\mathbf{X} \right)

.. math::
    = \frac{P}{2}trace\left( \left(\mathbf{X}^T\mathbf{V}^{-1}\mathbf{X} \right)^{-1}\mathbf{X}^T\mathbf{V}^{-1}\frac{\partial{\mathbf{V}}}{\partial{\theta_i}}\mathbf{V}^{-1}\mathbf{X} \right)

.. math::
    = \frac{P}{2}trace\left( \mathbf{V}^{-1}\mathbf{X}\left(\mathbf{X}^T\mathbf{V}^{-1}\mathbf{X} \right)^{-1}\mathbf{X}^T\mathbf{V}^{-1}\frac{\partial{\mathbf{V}}}{\partial{\theta_i}} \right)

Secondly, the derivative of the third term is

.. math::
    l=-\frac{1}{2}trace\left(\mathbf{V}_{R}^{-1}\mathbf{Y}\mathbf{Y}^T\right)

.. math::
    \frac{\partial{l}}{\partial{\theta_i}}=\frac{1}{2}trace\left( \mathbf{V}_{R}^{-1}\frac{\partial{\mathbf{V}}}{\partial{\theta_i}}\mathbf{V}_{R}^{-1}\mathbf{Y}\mathbf{Y}^T \right)

The last step is not easily proven, except for diligently applying the product rule and seeing a lot of terms cancel. Putting these two results together with the derivative of the normal likelihood gives us:

.. math::
    \frac{\partial(L_{ReML})}{\partial{\theta_i}}=-\frac{P}{2}trace\left( \mathbf{V}^{-1}\frac{\partial{\mathbf{V}}}{\partial{\theta_i}} \right)

.. math::
    + \frac{1}{2}trace\left(\mathbf{V}_{R}^{-1} \frac{\partial{\mathbf{V}}}{\partial{\theta_i}} \mathbf{V}_{R}^{-1} \mathbf{Y}\mathbf{Y}^T \right)

.. math::
    + \frac{P}{2}trace\left( \mathbf{V}^{-1}\mathbf{X}\left(\mathbf{X}^T\mathbf{V}^{-1}\mathbf{X} \right)^{-1}\mathbf{X}^T\mathbf{V}^{-1}\frac{\partial{\mathbf{V}}}{\partial{\theta_i}} \right)

.. math::
    =-\frac{P}{2}trace\left( \mathbf{V}_{R}^{-1} \frac{\partial{\mathbf{V}}}{\partial{\theta_i}} \right) + \frac{1}{2}trace\left(\mathbf{V}_{R}^{-1} \frac{\partial{\mathbf{V}}}{\partial{\theta_i}} \mathbf{V}_{R}^{-1} \mathbf{Y}\mathbf{Y}^T \right)

Derivates for specific parameters
---------------------------------

From the general term for the derivative of the log-likelihood, we can derive the specific expressions for each parameter. In general, we model the co-variance matrix of the data :math:`\mathbf{V}` as:

.. math::
    {\bf{V}}=s{\bf{ZG}}(\boldsymbol{\theta}_h){\bf{Z}}^{T}+S\sigma^{2}_{\epsilon}\\
    s=exp(\theta_{s})\\
    \sigma^2_{\epsilon} = exp(\theta_{\epsilon})

Where :math:`\theta_s` is the signal scaling parameter, the :math:`\theta_{\epsilon}` the noise parameter. We are using the exponential of the parameter, to ensure that the noise variance and the scaling will always be strictly positive. When taking the derivatives, we use the simple rule of :math:`\partial exp(x) / \partial x=exp(x)`.  Each model provides the partial derivaratives for :math:`\mathbf{G}` in respect to the model parameters (see above). From this we can easily obtain the derviative of :math:`\mathbf{V}`

.. math::
    \frac{\partial{\mathbf{V}}}{\partial{\theta_h}} = \mathbf{Z} \frac{\partial{\mathbf{G(\boldsymbol{\theta_h})}}}{\partial{\theta_h}}\mathbf{Z}^T exp(\theta_{s}).

The derivate in respect to the noise parameter

.. math::
    \frac{\partial{\mathbf{V}}}{\partial{\theta_{\epsilon}}} = \mathbf{S}exp(\theta_{\epsilon}).

And in respect to the signal scaling parameter

.. math::
    \frac{\partial{\mathbf{V}}}{\partial{\theta_{s}}} = {\bf{ZG}}(\boldsymbol{\theta}_h){\bf{Z}}^T exp(\theta_s).

Conjugate Gradient descent
--------------------------

One way of optiminzing the likelihood is simply using the first derviative and performing a conjugate-gradient descent algorithm. For this, the routines `pcm_likelihoodIndivid` and `pcm_likelihoodGroup` return the negative log-likelihood, as well as a vector of the first derivatives of the negative log-likelihood in respect to the parameter. The implementation of conjugate-gradient descent we are using here based on Carl Rassmussen's excellent  function `minimize`.

Newton-Raphson algorithm
------------------------

A alternative to conjugate gradients, which can be considerably faster, are optimisation routines that exploit the matrix of second derivatives of the log-liklihood. The local curvature information is then used to  "jump" to suspected bottom of the bowl of the likelihood surface. The negative expected second derivative of the restricted log-likelihood, also called Fisher-information can be calculated efficiently from terms that we needed to compute for the first derivative anyway:

.. math::
    {\mathbf{F}}_{i,j}(\theta) = - E \left[ \frac{\partial^2 }{\partial \theta_i \partial \theta_j} L_{ReML}\right]=\frac{P}{2}trace\left(\mathbf{V}^{-1}_{R} \frac{\partial \mathbf{V}}{\partial \theta_i}\mathbf{V}^{-1}_{R} \frac{\partial \mathbf{V}}{\partial \theta_j}  \right).

The update then uses a slightly regularized version of the second derviate to compute the next update on the parameters.

.. math::
    \boldsymbol{\theta}^{u+1}=\boldsymbol{\theta}^{u}-\left( \mathbf{F}(\boldsymbol{\theta}^{u})+{\mathbf{I}}\lambda\right)^{-1}\frac{\partial L_{ReML}}{\partial \boldsymbol{\theta}^{u}} .

Because the update can become  unstable, we are regularising the Fisher information matrix by adding a small value to the diagonal, similar to a Levenberg regularisation for least-square problems. If the likelihood increased,  :math:`\lambda` is decreases, if the liklihood accidentially decreased, then we take a step backwards and increase  :math:`\lambda`.  The algorithm is implemented in  `pcm_NR` .

Choosing an optimisation algorithm
----------------------------------
While the Newton-Raphson algorithm can be considerably faster for many problems, it is not always the case. Newton-Raphson usually arrives at the goal with many fewer steps than conjugate gradient descent, but on each step it has to calculate the matrix second derviatives, which grows in the square of the number of parameters . So for highly-parametrized models, the simple conjugate gradient algorithm is better. You can set for each model the desired algorithm by setting the field  `M.fitAlgorithm = 'NR';`  for Newton-Raphson and   `M.fitAlgorithm = 'minimize';` for conjugate gradient descent. If no such field is given, then fitting function will call `M=pcm_optimalAlgorithm(M)` to obtain a guess of what will be the best algorithm for the problem. While this function provides a good heuristic strategy, it is recommended to try both and compare both the returned likelihood and time. Small differences in the likelihood (:math:`<0.1`) are due to different stopping criteria and should be of no concern. Larger differences can indicate failed convergence.


Acceleration of matrix inversion
--------------------------------

When calculating the likelihood or the derviatives of the likelihood, the inverse of the variance-covariance has to be computed. Because this can become quickly very costly (especially if original time series data is to be fitted), we can  exploit the special structure of :math:`\mathbf{V}` to speed up the computation:

.. math::
    \begin{array}{c}{{\bf{V}}^{ - 1}} = {\left( {s{\bf{ZG}}{{\bf{Z}}^T} + {\bf{S}}\sigma _\varepsilon ^2} \right)^{ - 1}}\\ = {{\bf{S}}^{ - 1}}\sigma _\varepsilon ^{ - 2} - {{\bf{S}}^{ - 1}}{\bf{Z}}\sigma _\varepsilon ^{ - 2}{\left( {{s^{ - 1}}{\mathbf{G}^{ - 1}} + {{\bf{Z}}^T}{{\bf{S}}^{ - 1}}{\bf{Z}}\sigma _\varepsilon ^{ - 2}} \right)^{ - 1}}{{\bf{Z}}^T}{{\bf{S}}^{ - 1}}\sigma _\varepsilon ^{ - 2}\\ = \left( {{{\bf{S}}^{ - 1}} - {{\bf{S}}^{ - 1}}{\bf{Z}}{{\left( {{s^{ - 1}}{\mathbf{G}^{ - 1}}\sigma _\varepsilon ^2 + {{\bf{Z}}^T}{{\bf{S}}^{ - 1}}{\bf{Z}}} \right)}^{ - 1}}{{\bf{Z}}^T}{{\bf{S}}^{ - 1}}} \right)/\sigma _\varepsilon ^2 \end{array}

With pre-inversion of :math:`\mathbf{S}` (which can occur once outside of the iterations), we make a :math:`N{\times}N` matrix inversion into a `K{\times}K` matrix inversion.


