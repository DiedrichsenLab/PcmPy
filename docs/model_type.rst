.. _model_type:

Model types
===========
Independently of whether you choose an Encoding- or RSA-style approach to building your model, the PCM toolbox distinguishes between a number of different model types, each of which has an own model class.

Fixed models
------------
In fixed models, the second moment matrix :math:`\mathbf{G}` is exactly predicted by the model. The most common example is the Null model  :math:`\mathbf{G} = \mathbf{0}`. This is equivalent to assuming that there is no difference between any of the activity patterns. The Null-model is useful if we want to test whether there are any differences between experimental conditions. An alternative Null model would be :math:`\mathbf{G} = \mathbf{I}`, i.e. to assume that all patterns are uncorrelated equally far away from each other.

Fixed models also occur when the representational structure can be predicted from some independent data. An example for this is shown in the following example, where we predict the structure of finger representations directly from the correlational structure of finger movements in every-day life (Ejaz et al., 2015). Importantly, fixed models only predict the the second moment matrix up to a proportional constant. The width of the distribution will vary with the overall scale or signal-to-noise-level. Thus, when evaluating fixed models we usually allow the predicted second moment matrix to be scaled by an arbitrary positive constant (see :ref:`fitting`).

Example
^^^^^^^
An empirical example to for a fixed representational model comes from Ejaz et al (2015). Here the representational structure of 5 finger movements was compared to the representational structure predicted by the way the muscles are activated during finger movements (Muscle model), or by the covariance structure of natural movements of the 5 fingers. That is the predicted second moment matrix is derived from data completely independent of our imaging data.

Models are a specific class, inherited from the class ``Model``. To define a fixed model, we simple need to load the predicted second moment matrix and define a model structure as follows (see :ref:`examples`):

.. sourcecode:: python

    M1 = pcm.model.FixedModel('null',np.eye(5))    # Makes a Null model
    M2 = pcm.model.FixedModel('muscle',modelM[0])  # Makes the muscle model
    M3 = pcm.model.FixedModel('natural',modelM[1]) # Makes the natural stats model
    M = [M1, M2, M3] # Join the models for fitting in list

When evaluating the likelihood of a data set under the prediction, the pcm toolbox still needs to estimate the scaling factor and the noise variance, so even in the case of fixed models, an iterative maximization of the likelihood is required (see below).

Component models
----------------

A more flexible model is to express the second moment matrix as a linear combination of different components. For example, the representational structure of activity patterns in the human object recognition system in inferior temporal cortex can be compared to the response of a convolutional neural network that is shown the same stimuli (Khaligh-Razavi & Kriegeskorte, 2014). Each layer of the network predicts a specific structure of the second moment matrix and therefore constitutes a fixed model. However, the representational structure may be best described by a mixture of multiple layers. In this case, the overall predicted second moment matrix is a linear sum of the weighted components matrices:

.. math::
    \mathbf{G}= \sum_{h}{\exp(\theta_{h})\mathbf{G}_{h}}

The weights for each component need to be positive - allowing negative weights would not guarantee that the overall second moment matrix would be positive definite. Therefore we use the exponential of the weighing parameter here, such that we can use unconstrained optimization to estimate the parameters.

For fast optimization of the likelihood, we require the derivate of the second moment matrix in respect to each of the parameters. Thus derivative can then be used to calculate the derivative of the log-likelihood in respect to the parameters (see section 4.3. Derivative of the log-likelihood). In the case of linear component models this is easy to obtain.

.. math::
    \frac{\partial \mathbf{G}}{\partial {\theta }_{h}}=\exp(\theta_{h}) {\bf{G}}_{h}


Example
^^^^^^^

In the example :ref:`Component model <examples>`, we have two fixed models, the Muscle and the natural statistics model. One question that arises in the paper is whether the real observed structure is better fit my a linear combination of the natural statistics and the muscle activity structure. So we can define a third model, which allows any arbitrary mixture between the two type.

.. sourcecode:: python

    MC = pcm.ComponentModel('muscle+nat',[modelM[0],modelM[1]])

Feature models
--------------

A representational model can be also formulated in terms of the features that are thought to be encoded in the voxels. Features are hypothetical tuning functions, i.e.\ models of what activation profiles of single neurons could look like. Examples of features would be Gabor elements for lower-level vision models, elements with cosine tuning functions for different movement directions for models of motor areas, and semantic features for association areas. The actual activity profiles of each voxel are a weighted combination of the feature matrix :math:`\mathbf{u}_p = \mathbf{M} \mathbf{w}_p`. The predicted second moment matrix of the activity profiles is then :math:`\mathbf{G} = \mathbf{MM}^{T}`, assuming that all features are equally strongly and independently encoded, i.e. :math:`E \left(\mathbf{w}_p\mathbf{w}_p^{T} \right)=\mathbf{I}`. A feature model can now be flexibly parametrized by expressing the feature matrix as a weighted sum of linear components.

.. math::
    \mathbf{M}= \sum_{h} \theta_h \mathbf{M}_{h}


Each parameter :math:`\theta_h` determines how strong the corresponding set of features is represented across the population of voxels. Note that this parameter is different from the actual feature weights :math:`\mathbf{W}`. Under this model, the second moment matrix becomes

.. math::
    \mathbf{G}=\mathbf{UU}^{T}/P=\frac{1}{P}\sum_{h}\theta_{h}^{2}\mathbf{M}_{h}\mathbf{M}_{h}^{T}+\sum_{i}\sum_{j}\theta_{i}\theta_{j}\mathbf{M}_{i}\mathbf{M}_{j}^{T}.

From the last expression we can see that, if features that belong to different components are independent of each other, i.e. :math:`\mathbf{M}_{i} \mathbf{M}_{j} = \mathbf{0}`, then a feature model is equivalent to a component model with :math:`\mathbf{G}_h = \mathbf{M}_{h}\mathbf{M}_{h}^{T}`.  The only technical difference is that we use the square of the parameter :math:`\theta_h`, rather than its exponential, to enforce non-negativity. Thus, component models assume that the different features underlying each component are encoded independently in the population of voxels - i.e.\ knowing something about the tuning to feature of component A does not tell you anything about the tuning to a feature of component B. If this cannot be assumed, then the representational model is better formulated as a feature model.

By the product rule for matrix derivatives, we have

.. math::
    \frac{{\partial {\bf{G}}}}{{\partial {\theta_h}}} = {{\bf{M}}_h}{\bf{M}}{\left( \bf{\theta} \right)^T} + {\bf{M}}\left( \theta \right){\bf{M}}_h^T


Correlation model
-----------------

The correlation model class is designed model correlation between specific sets of activity patterns. This problem often occurs in neuroimaging studies: For example, we may have  3 actions that are measured under two conditions (for example observation and execution), and we want to know to what degree the activity patterns of observing an action related to the pattern observed when executing the same action.

To estimate the maximum-likelihood estimate of the correlation, we need to estimate the variance of the activity patterns within each condition (which is positive), as well as the correlation (which is between -1 and 1). We are therefore using the following nonlinear transforms to ensure that the parameters are unconstrained:

.. math::
    \sigma^2_x = exp(\theta_x)\\
    \sigma^2_y = exp(\theta_y)\\
    \r = (exp(2*\theta_z)-1)/(exp(2*\theta_z)+1) 

THe overall second moment matrix is then given by: 

.. math::
    \mathbf{G}= \begin{bmatrix}
    \sigma^2_x & r \sigma_x \sigma_y \\
    r \sigma_x \sigma_y & \sigma^2_y
    \end{bmatrix}\\

If you have more than one item in each condition, the model will assume that the items within condition have independent activity patterns. If this is not the case, then you can also provide a item x item matrix as the argument  `within_cov` as . For multiple items we usually want to ignore any overall differences between the conditions, so `cond_effect` needs to be set to be `True`. 

The derivatives are automatically calculated in the predict function. Specfically, the derivative of the correlation parameters is easy to obtain:

.. math::
    r=\frac{exp(2\theta)-1}{exp(2\theta)+1}\\
    \theta=\frac{1}{2}log\left(\frac{1+\theta}{1-\theta}\right)\\

The derivative of :math:`r` in respect to :math:`\theta` can be derived using the product rule:

.. math::
    \frac{\partial r}{\partial \theta} =
    \frac{2 exp(2 \theta)}{exp(2\theta)+1} - \frac{\left(exp(2\theta)-1\right)\left(2 exp(2 \theta)\right)}{\left(exp(2\theta)+1\right)^2} = \\
    \frac{4 exp(2 \theta)}{\left(exp(2\theta)+1\right)^2}

Example
^^^^^^^
For a full example, please see the :ref:`Correlation model <example>`.

:doc:`../docs/demos/demo_correlation.ipynb`

Free models
-----------
The most flexible representational model is the free model, in which the predicted second moment matrix is unconstrained. Thus, when we estimate this model, we would simply derive the maximum-likelihood estimate of the second-moment matrix. This model is mainly useful if we want to obtain an estimate of the maximum likelihood that could be achieved with a fully flexible model, i.e the noise ceiling (Nili et al. 20).

In estimating an unconstrained :math:`\mathbf{G}`, it is important to ensure that the estimate will still be a positive definite matrix. For this purpose, we express the second moment as the square of an upper-triangular matrix, :math:`\mathbf{G} = \mathbf{AA}^{T}` (Diedrichsen et al., 2011; Cai et al., 2016). The parameters are then simply all the upper-triangular entries of :math:`\mathbf{A}`.

Example
^^^^^^^
To set up a free model, simple create a new model of type ``FreeModel``.

.. sourcecode:: python

    M5 = pcm.model.FreeModel('ceil',n_cond)

If the number of conditions is very large, the crossvalidated estimation of the noise ceiling model can get rather slow. For a quick and approximate noise ceiling, you can also set use an unbiased estimate of the second moment matrix from ``pcm.util.est_G_crossval`` to determine the parameters - basically the starting values of the complete model. This will lead to slightly lower noise ceilings as compared to the full optimization, but large improvements in speed.

Custom model
------------

In some cases, the hypotheses cannot be expressed by a model of the type mentioned above. Therefore, the PCM toolbox allows the user to define their own custom model. In general, the predicted second moment matrix is a non-linear (matrix valued) function of some parameters, :math:`\mathbf{G}=F\left(\theta\right)`. One example is a representational model in which the width of the tuning curve (or the width of the population receptive field) is a free parameter. Such parameters would influence the features, and hence also the second-moment matrix in a non-linear way. Computationally, such non-linear models are not much more difficult to estimate than component or feature models, assuming that one can analytically derive the matrix derivatives :math:`\partial \mathbf{G} / \partial \theta_{h}`.

To define a custom model, the user needs to define a new Model class, inherited from the abstract class ``pcm.model.Model``. The main thing is to define the ``predict`` function, which takes the parameters as an input and returns **G** the partial derivatives of **G** in respect to each of these parameters. The derivates are returned as a (HxKxK) tensor, where H is the number of parameters.

.. sourcecode:: python

    class CustomModel(Model):
    # Constructor of the class
    def __init__(self,name,...):
        Model.__init__(self,name)
        ...

    # Prediction function
    def predict(self,theta):
        G = .... # Calculate second momement matrix
        dG_dTheta = # Calculate derivative second momement matrix
        return (G,dG_dTheta)

    #  Intiialization function
    def set_theta0(self,G_hat):
        """
        Sets theta0 based on the crossvalidated second-moment

        Parameters:
            G_hat (numpy.ndarray)
                Crossvalidated estimate of G
        """
        # The function can use G_hat to get good starting values,
        # or just start at fixed values
        self.theta0 = ....


Note that the predict function is repeatedly called by the optimization routine and needs to execute fast. That is, any computation that does not depend on the current value of :math:`\theta` should be performed outside the function and stored in the object.
