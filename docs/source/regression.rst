.. _regression:

Regularized regression
======================
PCM can be used to tune the regularization parameter for ridge regression. Specifically, ridge regression is a special case of the PCM model

.. math::
    \mathbf{y_i} = \mathbf{Z} \mathbf{u}_i + \mathbf{X} \boldsymbol{\beta}_i +  \boldsymbol{\epsilon}_p

where :math:`\mathbf{u} \sim N(0,\mathbf{I} s)` are the vectors of  random effects, and :math:`\boldsymbol{\epsilon} \sim N(0,\mathbf{I} \sigma^2_{\epsilon})` the measurement error.

:math:`\boldsymbol{\beta}` are the fixed effects - in the case of standard ridge regression, is the intercept. In this case :math:`\mathbf{X}` would be a vector of 1s. The more general implementation allows arbitrary fixed effects, which may also be correlated with the random effects.

Assuming that the intercept is already removed, the random effect estimates are:

.. math::
    \begin{array}{c}
    \hat{\mathbf{u}} = (\mathbf{Z}^T \mathbf{Z} + \mathbf{I} \lambda)^{-1} \mathbf{Z}^T \mathbf{y}_i \\
    \lambda = \frac{\sigma^2_{\epsilon}}{s} = \frac{exp(\theta_s)}{exp(\theta_{\epsilon})}
    \end{array}

This makes the random effects estiamtes in PCM indentical to Ridge regression with an optimal regularization coefficient :math:`\lambda`.

The PCM regularized regression model is design to work with multivariate data, i.e. many variables :math:`\mathbf{y}_i, \ddots, \mathbf{y}_P` that all share the same generative model (:math:`\mathbf{X}`, :math:`\mathbf{Z}`), but have different random and fixed effects. Of course, the regression model works on univariate regression models with only one data vector.

Most importantly, the pcm regression model allows you to estimate a different ridge coefficients for different columns of the design matrix. In general, we can can set the covariance matrix of :math:`\mathbf{u}` to

.. math::
    G =   \begin{bmatrix}
    exp(\theta_1) & &  &\\
    & exp(\theta_1) &  &\\
    & &  \ddots        &\\
    & & &   exp(\theta_Q)
    \end{bmatrix}

where Q groups of effects share the same variance (and therefore the same Ridge coefficient). In the extreme, every column in the design matrix would have its own regularization parameter to be estimated. The use of the Restricted Maxmimum Likelihood (ReML) makes the estimation of such more complex regularisation both stable and computationally feasible.

Practical guide
---------------

See Jupyter notebook ``demos/demo_regression.ipynb`` for a working example, which also shows a direct comparison to ridge regression. In this work book, we generate a example with `N = 100` observations, `P = 10` variables, and `Q = 10` regressors:

.. sourcecode:: python

    # Make the training data:
    N = 100 # Number of observations
    Q = 10  # Number of random effects regressors
    P = 10  # Number of variables
    Z = np.random.normal(0,1,(N,Q)) # Make random design matrix
    U = np.random.normal(0,1,(Q,P))*0.5 # Make random effects
    Y = Z @ U + np.random.normal(0,1,(N,P)) # Generate training data
    # Make testing data:
    Zt = np.random.normal(0,1,(N,Q))
    Yt = Zt @ U + np.random.normal(0,1,(N,P))

Given this data, we can now define Ridge regression model, where all regressors are sharing the same ridge coefficient. :code:`comp` is a index matrix for each column of *Z*.

.. sourcecode:: python

    # Vector indicates that all columns are scaled by the same parameter
    comp = np.array([0,0,0,0,0,0,0,0,0,0])
    # Make the model
    M1 = pcm.regression.RidgeDiag(comp, fit_intercept = True)
    # Estimate optimal regularization parameters from training data
    M1.optimize_regularization(Z,Y)

After estimation, the two theta parameters (for signal and noise) can be retrieved from :code:`M1.theta_`. The Regularization parameter for ridge regression is then :code:`exp(M1.theta_[1])/exp(M1.theta_[0])`.

The model then can be fitted to the training (or other) data to determine the coefficients.

.. sourcecode:: python

    # Addition fixed effects can be passed in X
    M1.fit(Z, Y, X = None)

The random effect coefficients are stored in :code:`M1.coefs_` and the fixed effects in :code:`M1.beta_s`.

Finally we can predict the data for he indepenent test set and evaluate this predicton.

.. sourcecode:: python

    Yp = M1.predict(Zt)
    R2 = 1- np.sum((Yt-Yp)**2)/np.sum((Yt)**2)

Finally, if we want to estimate the importance of different groups of columns, we can define different ridge coefficients for different groups of columns:

.. sourcecode:: python

    comp = np.array([0,0,1,1,1,1,1,2,2,2])
    M2 = pcm.regression.RidgeDiag(comp, fit_intercept = True)

In this example, the first 2, the next 5, and the last 3 columns share one Ridge coefficient. The call to :code:`M1.optimize_regularization(Z,Y)` causes 4 theta parameters and hence 3 regularization coefficients to be estimated. If the importance of different columns of the design matrix is truely different, this will provide better predictions.

PCM vs. GridSearchCV
--------------------

In terms of accuracy

In terms of speed,

Scaling in N

Scaling in P

Scaling in number of components


