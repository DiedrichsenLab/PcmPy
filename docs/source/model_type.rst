.. _model_type:

Model types
===========
Independently of whether you choose a Encoding- or RSA-style approach to arrive at the final model, the PCM toolbox distinguishes between a number of different model types.  

Fixed models
------------
In fixed models, the second moment matrix :math:`\mathbf{G}` is exactly predicted by the model. The simplest and most common example is the Null model, which states that :math:`\mathbf{G} = \mathbf{0}`. This is equivalent to assuming that there is no difference between the activity patterns measured under any of the conditions. The Null-model is useful if we want to test whether there are any differences between experimental conditions.

Fixed models also occur when the representational structure can be predicted from some independent data. An example for this is shown in the following example, where we predict the structure of finger representations directly from the correlational structure of finger movements in every-day life [@RN3415]. Importantly, fixed models only predict the the second moment matrix up to a proportional constant. The width of the distribution will vary with the overall signal-to-noise-level (assuming we use pre-whitened data). Thus, when evaluating fixed models we allow the predicted second moment matrix to be scaled by an arbitrary positive constant.

Example
^^^^^^^
An empirical example to for a fixed representational model comes from Ejaz et al (2015). Here the representational structure of 5 finger movements was compared to the representational structure predicted by the way the muscles are activated during finger movements (Muscle model), or by the covariance structure of natural movements of the 5 fingers. That is the predicted second moment matrix is derived from data completely independent of our imaging data.

Models are stored in structures, with the field `type` indicating the model type. To define a fixed model, we simple need to load the predicted second moment matrix and define a model structure as follows (see `pcm_recipe_finger`): 

.. sourcecode:: python

    M.numGparams = 0; % Number of parameters
    M.Gc = Model(1).G_cent; % This is the predicted second moment matrix from the behavioural data
    M.name = 'muscle’;% This is the name of the

When evaluating the likelihood of a data set under the prediction, the pcm toolbox still needs to estimate the scaling factor and the noise variance, so even in the case of fixed models, an iterative maximization of the likelihood is required (see below).

Component models
----------------

A more flexible model is to express the second moment matrix as a linear combination of different components. For example, the representational structure of activity patterns in the human object recognition system in inferior temporal cortex can be compared to the response of a convolutional neural network that is shown the same stimuli [@RN3544]. Each layer of the network predicts a specific structure of the second moment matrix and therefore constitutes a fixed model. However, the real representational structure seems to be best described by a mixture of multiple layers. In this case, the overall predicted second moment matrix is a linear sum of the weighted components matrices:


.. math::
    \mathbf{G}= \sum_{h}{\exp(\theta_{h})\mathbf{G}_{h}}

The weights for each component need to be positive - allowing negative weights would not guarantee that the overall second moment matrix would be positive definite. Therefore we use the exponential of the weighing parameter here, such that we can use unconstrained optimization to estimate the parameters.

For fast optimization of the likelihood, we require the derivate of the second moment matrix in respect to each of the parameters. Thus derivative can then be used to calculate the derivative of the log-likelihood in respect to the parameters (see section 4.3. Derivative of the log-likelihood). In the case of linear component models this is easy to obtain.

.. math::
    \frac{\partial \mathbf{G}}{\partial {\theta }_{h}}=\exp(\theta_{h}) {\bf{G}}_{h}


Example
^^^^^^^

In the example `pcm_recipe_finger`, we have two fixed models, the Muscle and the natural statistics model. One question that arises in the paper is whether the real observed structure is better fit my a linear combination of the natural statistics and the muscle activity structure. So we can define a third model, which allows any arbitrary mixture between the two type.

.. sourcecode:: python

    M.type = ‘component’;
    M.numGparams = 2;
    M.Gc(:,:,1) = Model(1).G_cent;
    M.Gc(:,:,2) = Model(2).G_cent;
    M.name = 'muscle + usage';

Feature models
--------------

A representational model can be also formulated in terms of the features that are thought to be encoded in the voxels. Features are hypothetical tuning functions, i.e.\ models of what activation profiles of single neurons could look like. Examples of features would be Gabor elements for lower-level vision models [@RN3098], elements with cosine tuning functions for different movement directions for models of motor areas [@RN2960], and semantic features for association areas [@RN3566]. The actual activity profiles of each voxel are a weighted combination of the feature matrix :math:`\mathbf{u}_p = \mathbf{M} \mathbf{w}_p`. The predicted second moment matrix of the activity profiles is then :math:`\mathbf{G} = \mathbf{MM}^{T}`, assuming that all features are equally strongly and independently encoded, i.e. :math:`E \left(\mathbf{w}_p\mathbf{w}_p^{T} \right)=\mathbf{I}`. A feature model can now be flexibly parametrized by expressing the feature matrix as a weighted sum of linear components.

.. math::
    \mathbf{M}= \sum_{h} \theta_h \mathbf{M}_{h}


Each parameter :math:`\theta_h` determines how strong the corresponding set of features is represented across the population of voxels. Note that this parameter is different from the actual feature weights :math:`\mathbf{W}`. Under this model, the second moment matrix becomes

.. math::
    \mathbf{G}=\mathbf{UU}^{T}/P=\frac{1}{P}\sum_{h}\theta_{h}^{2}\mathbf{M}_{h}\mathbf{M}_{h}^{T}+\sum_{i}\sum_{j}\theta_{i}\theta_{j}\mathbf{M}_{i}\mathbf{M}_{j}^{T}.

From the last expression we can see that, if features that belong to different components are independent of each other, i.e. :math:`\mathbf{M}_{i} \mathbf{M}_{j} = \mathbf{0}`, then a feature model is equivalent to a component model with :math:`\mathbf{G}_h = \mathbf{M}_{h}\mathbf{M}_{h}^{T}`.  The only technical difference is that we use the square of the parameter :math:`\theta_h`, rather than its exponential, to enforce non-negativity. Thus, component models assume that the different features underlying each component are encoded independently in the population of voxels - i.e.\ knowing something about the tuning to feature of component A does not tell you anything about the tuning to a feature of component B. If this cannot be assumed, then the representational model is better formulated as a feature model. 

By the product rule for matrix derivatives, we have

.. math::
    \frac{{\partial {\bf{G}}}}{{\partial {\theta_h}}} = {{\bf{M}}_h}{\bf{M}}{\left( \bf{\theta} \right)^T} + {\bf{M}}\left( \theta \right){\bf{M}}_h^T

Example
^^^^^^^ 
In the example `pcm_recipe_feature`, we want to model the correlation between the patterns for the left hand and the corresponding fingers for the right hand. 

![*Feature model to model correlation.*](Figures/Figure_feature_corr.pdf){#fig:Fig2}

There two features to simulate the common pattern for the left and right hand movements, respectively (:math:`\theta_{d}`, :math:`\theta_{e}`). For the fingers of the contra-lateral hand we have one feature for each finger, with the feature component weighted by :math:`\theta_{a}`. The same features also influence the patterns for the ipsilateral hand with weight :math:`\theta_{b}`. This common component models the correspondence between contra and ipsilateral fingers. Finally, the component weighted by :math:`\theta_{c}` encodes unique encoding for the ipsilateral fingers.  

.. sourcecode:: python

    M.type       = 'feature';
    M.numGparams = 5;
    M.Ac(:,1:5 ,1)  = [eye(5);zeros(5)];      % Contralateral finger patterns   (a)
    M.Ac(:,1:5 ,2)  = [zeros(5);eye(5)];      % Mirrored Contralateralpatterns  (b)
    M.Ac(:,6:10,3)  = [zeros(5);eye(5)];      % Unique Ipsilateral pattterns    (c)
    M.Ac(:,11  ,4)  = [ones(5,1);zeros(5,1)]; % Hand-specific component contra  (d)
    M.Ac(:,12  ,5)  = [zeros(5,1);ones(5,1)]; % Hand-specific component ipsi    (e)
    M.name       = 'correlation';
    M.theta0=[1 1 0.5 0.1 0.1 ]';		% Starting values 

Nonlinear models
----------------

The most flexible way of defining a representational model is to express the second moment matrix as a non-linear (matrix valued) function of the parameters, :math:`\mathbf{G}=F\left(\theta\right)`. While often a representational model can be expressed as a component or feature model, sometimes this is not possible. One example is a representational model in which the width of the tuning curve (or the width of the population receptive field) is a free parameter [@RN3558]. Such parameters would influence the features, and hence also the second-moment matrix in a non-linear way. Computationally, such non-linear models are not much more difficult to estimate than component or feature models, assuming that one can analytically derive the matrix derivatives :math:`\partial \mathbf{G} / \partial \theta_{h}`. 

For this, the user needs to define a function that takes the parameters as an input and returns **G** the partial derivatives of **G** in respect to each of these parameters. The derivates are returned as a (KxKxH) tensor, where H is the number of parameters. 

.. sourcecode:: python

    [G,dGdtheta]=fcn(theta,M)

Note that this function is repeatedly called by the optimization routine and needs to execute fast. That is, any computation that does not depend on the current value of :math:`\theta` should be performed outside the function and then passed to it.

Example 1: Nonlinear scaling model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
In the example `pcm_recipe_nonlinear`, we define how the representational structure of single finger presses of the right hand (**G**) scales as the number of presses increases. To achieve this, we can simply allow for a scaling component (:math:`\theta_{f}`) for each pressing speed (*f*). In the recipe, we have four pressing speeds. Therefore, we use **G** from one pressing speed to model the **G**s of the remaining three pressing speeds. For one pressing speed, **G** is a 5x5 matrix, where each dimension corresponds to one finger. To speed up the optimization routine, we set :math:`\,mathbf{G}(1,1)` to one. The parameters in **G** are then free to vary with respect to :math:`\,mathbf{G}(1,1)`. 

.. sourcecode:: python

    M.type       = 'nonlinear'; 
    M.name       = 'Scaling';
    M.modelpred  = @ra_modelpred_scale;	
    M.numGparams = 17; 					% 14 free theta params in G because G(1,1) is set to 1, and 3 free scaling params
    M.theta0     = [Fx0; scale_vals];   % Fx0 are the 14 starting values from G, scale_vals are 3 starting scaling values 

Example 2: Nonlinear correlation model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
In the example `pcm_recipe_correlation`, we use a non-linear model to determine the correlation between two sets of 5 patterns corresponding to 5 items (e.g. motor sequences) measured under two conditions (e.g. two testing sessions). We use two approaches: 

**Fixed correlation models**: We use a series of 21 models that test the likelihood of the data under a fixed correlations between -1 and 1. This approach allows us to determine how much evidence we have for one specific correlation over the other. Even though the correlation is fixed for these models, the variance structure within each of the conditions is flexibly estimated. This is done using a compent model within each condition. 

.. math::
    \mathbf{G}^{(1)} = \sum_{h}{\exp(\theta^{(1)}_{h})\mathbf{G}^{(1)}_{h}}\\
    \mathbf{G}^{(2)} = \sum_{h}{\exp(\theta^{(2)}_{h})\mathbf{G}^{(2)}_{h}}\\

The overall model is nonlinear, as the two components interact in the part of the **G** matrix that indicates the covariance between the patterns of the two conditions (**C**). Given a constant correlation *r*, the overall second moment matrix is calculated as:

.. math::
    \mathbf{G}= \begin{bmatrix} 
    \mathbf{G}^{(1)} & r\mathbf{C} \\
    r\mathbf{C}^T & \mathbf{G}^{(2)}
    \end{bmatrix}\\
    \mathbf{C}_{i,j} = \sqrt{\mathbf{G}^{(1)}_{i,j}\mathbf{G}^{(2)}_{i,j}}

The derivatives of that part of the matrix in respect to the parameters :math:`\theta^{(1)}_{h}` then becomes 

.. math::
    \frac{{\partial {\mathbf{C}_{i,j}}}}{{\partial {\theta^{(1)}_h}}} =
    \frac{r}{2 \mathbf{C}_{i,j}} \mathbf{G}^{(2)}_{i,j} \frac{{\partial {\mathbf{G}^{(1)}_{i,j}}}}{{\partial {\theta^{(1)}_h}}}

These derivatives are automatically calculated in the function `pcm_calculateGnonlinCorr`. From the log-likelihoods for each model, we can then obtain an approximation for the posterior distribution.  The models with a fixed correlation for our example can be generated using 

.. sourcecode:: python

    nModel  = 21; 
    r = linspace(-1,1,nModel); 
    for i=1:nModel             
    ​    M{i} = pcm_buildCorrModel('type','nonlinear','withinCov','individual','numItems',5,'r',r(i)); 
    end

**Flexible correlation model**: We also use a flexible correlation model, which has an additional model parameter for the correlation. To avoid bounds on the correlation, this parameter is the inverse Fisher-z transformation of the correlation, which can take values of :math:`[-\infty,\infty]`. 

.. math::
    \theta=\frac{1}{2}log\left(\frac{1+\theta}{1-\theta}\right)\\
    r=\frac{exp(2\theta)-1}{exp(2\theta)+1}\\

The derivative of :math:`r` in respect to :math:`\theta` can be derived using the product rule: 

.. math::
    \frac{\partial r}{\partial \theta} = 
    \frac{2 exp(2 \theta)}{exp(2\theta)+1} - \frac{\left(exp(2\theta)-1\right)\left(2 exp(2 \theta)\right)}{\left(exp(2\theta)+1\right)^2} = \\
    \frac{4 exp(2 \theta)}{\left(exp(2\theta)+1\right)^2}

Again, this derivative is automatically calculated by  `pcm_calculateGnonlinCorr` if `M.r` is set to `'flexible'`. 

.. sourcecode:: python

    Mf = pcm_buildCorrModel('type','nonlinear','withinCov','individual','numItems',5,'r','flexible'); 

Free models
-----------
The most flexible representational model is the free model, in which the predicted second moment matrix is unconstrained. Thus, when we estimate this model, we would simply derive the maximum-likelihood estimate of the second-moment matrix. This can be useful for a number of reasons. First, we may want an estimate of the second moment matrix to derive the corrected correlation between different patterns, which is less influenced by noise than the simple correlation estimate [@RN3638; @RN3033]. Furthermore, we may want to estimate the likelihood of the data under a free model to obtain a noise ceiling - i.e.\ an estimate of how well the best model should fit the data (see section Noise Ceilings).

In estimating an unconstrained :math:`\mathbf{G}`, it is important to ensure that the estimate will still be a positive definite matrix. For this purpose, we express the second moment as the square of an upper-triangular matrix, :math:`\mathbf{G} = \mathbf{AA}^{T}` [@RN3638; @RN3033]. The parameters are then simply all the upper-triangular entries of :math:`\mathbf{A}`.

Example
^^^^^^^
To set up a free model, the model type needs to be set to `freechol`, and you need to provide the number of conditions. The function `pcm_prepFreeModel` then quickly calculates the row and column indices for the different free parameters, which is a useful pre-computation for subsequent model fitting. 

.. sourcecode:: python

    M.type       = 'freechol'; 
    M.numCond    = 5;
    M.name       = 'noiseceiling'; 
    M            = pcm_prepFreeModel(M); 

For a quick and approximate noise ceiling, you can also set the model type to `freedirect`. In the case, the fitting algorithms simply use the crossvalidated second moment to determine the parameters - basically the starting values of the complete model. This may lead to a slightly lower noise ceiling, as full optimization is avoided in the interest of speed.