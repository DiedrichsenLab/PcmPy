.. _model_statistical:

Statistical Model
-----------------
PCM is based on a generative model of the measured brain activity data **Y**, a matrix of *N x P* activity measurements, referring to *N* time points (or trials) and *P* voxels (or channels). The data can refer to the minimally preprocessed raw activity data, or to already deconvolved activity estimates, such as those obtained as beta weights from a first-level time series model. **U** is the matrix of true activity patterns (a number of conditions x number of voxels matrix) and **Z** the design matrix. Also influencing the data are effects of no interest **B** and noise:

.. math::
    \begin{array}{c}\mathbf{Y} = \mathbf{ZU+XB}+\epsilon\\
    \mathbf{u}_{p}  \sim N(\mathbf{0},\mathbf{G})\\
    \epsilon_p \sim N(\mathbf{0},\mathbf{S}\sigma^{2}) \end{array}

Assumption about the signal (**U**)
...................................
The activity profiles (:math:`\mathbf{u}_p` columns of **U**) are considered to be a random variable. PCM models do not specify the exact activity profiles of specific voxels, but rather their probability distribution. Also, PCM is not interested in how the different activity profiles are spatially arranged. This makes sense considering that activity patterns can vary widely across different participants and do not directly impact what can be decoded from a region. For this, only the distribution of activity profiles in a region is important.

PCM assumes is that the expected mean of the activity profiles is zero. In many cases, we are not interested in how much a voxel is activated, but only how acitivity differs between conditions. In these cases, we model the mean for each voxel using the fixed effects :math:`\mathbf{X}`. 

Note that this mean pattern removal does not change in information contained in a region. In contrast, sometimes researchers also remove the mean value (Walther et al., 2016), i.e., the mean of each condition across voxels. We discourage this approach, as it would remove differences that, from the persepctive of decoding and representation, are highly meaningful.

The third assumption is that the activity profiles come from a multivariate Gaussian distribution. This is likely the most controversial assumption, but it is motivated by a few reasons: First, for fMRI data the multi-variate Gaussian is often a relatively appropriate description. Secondly, the definition causes us to focus on the mean and covariance matrix, **G**, as sufficient statistics, as these completely determine the Gaussian. Thus, even if the true distribution of the activity profiles is better described by a non-Gaussian distribution, the focus on the second moment is sensible as it characterizes the linear decodability of any feature of the stimuli.

Assumptions about the Noise
...........................

We assume that the noise of each voxel is Gaussian with a temporal covariance that is known up to a constant term :math:`\sigma^{2}`. Given the many additive influences of various noise sources on fMRI signals, Gaussianity of the noise is, by the central limit theorem, most likely a very reasonable assumption, commonly made in the fMRI literature. The original formulation of PCM used a model which assumed that the noise is also temporally independent and identically distributed (i.i.d.) across different trials, i.e. :math:`\mathbf{S} = \mathbf{I}` . However, as pointed out recently (Cai et al., 2016), this assumption is often violated in non-random experimental designs with strong biasing consequences for estimates of the covariance matrix. If this is violated, we can either assume that we have a valid estimate of the true covariance structure of the noise (**S**), or we can model different parts of the noise structure (see :ref:`model_noise`).

PCM also assumes that different voxels are independent from each other. If we use fMRI data, this assumption would be clearly violated, given the strong spatial correlation of noise processes in fMRI. To reduce these dependencies we typically uses spatially pre-whitened data, which is divided by a estimate of the spatial covariance matrix (Walther et al., 2016). Recent result from our lab show that this approach is sufficient to obtain correct marginal likelihoods. 

Marginal likelihood
...................

When we fit a PCM model, we are not trying to estimate specific values of the the estimates of the true activity patterns **U**. This is a difference to encoding approaches, in which we would estimate the values of :math:`\mathbf{U}` by estimating the feature weights :math:`\mathbf{W}`. Rather, we want to assess how likely the data is under any possible value of **U**, as specified by the prior distribution. Thus we wish to calculate the marginal likelihood

.. math::
    p\left(\mathbf{Y}|\theta\right)=\int p\left(\mathbf{Y}|\mathbf{U},\theta\right) p\left(\mathbf{U}|\theta\right) d\mathbf{U}.

This is the likelihood that is maximized in PCM in respect to the model parameters :math:`\theta`. For more details, see mathematical and algorithmic details.
