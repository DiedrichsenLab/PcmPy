Model Fitting
=============
Details of the different optimization routines that maximize the likelihood can be found in section on Mathematical and Algorithmic details. Currently, the toolbox either uses `minimize` (conjugate gradient descent) or `pcm_NR` (Newton-Raphson). Newton-Raphson can be substantially faster in cases in which there are relatively few free parameters.  The optimisation routine can be set for each model seperately by setting the field `M.fitAlgorithm`. 

The following routines are wrapper functions around the actual optimization routines that fit models to individual or group data. Noise and (possibly) scale parameters are added to the fit for each subject. To compare models of different complexity, 2 types of crossvalidation are implemented. 

![*Model crossvalidation schemes.* (**A**) *Within-subject crossvalidation where the model is fit on N-1 partitions and then evaluated on the left-out partition N.* (**B**) *Group crossvalidation were the model is fit to N-1 subjects and then evaluated on a left-out subject. In all cases, an individual scaling and noise parameters is fit to each subject to allow for different signal-to-noise levels.* ](Figures/Figure_crossval.pdf){#fig:Fig3}

Fitting to individual data sets
-------------------------------
Models can be fitted to each data set individually, using the function `pcm_fitModelIndivid`. Individual fitting makes sense for models with a single component (fixed models), which can be evaluated without crossvalidation. It also makes sense when the main interest are the parameters of the fit from each individual. 

.. sourcecode::python

   function [T,theta_hat,G_pred]=pcm_fitModelIndivid(Y,M,partitionVec,conditionVec,varargin);

The input arguments are: 

.. sourcecode::python

   % INPUT:
   %        Y: {#Subjects}.[#Conditions x #Voxels]
   %            Observed/estimated beta regressors from each subject.
   %            Preferably multivariate noise-normalized beta regressors.
   %
   %        M: {#Models} Cell array with structure that defines model(s). 
   %
   %   partitionVec: {#Subjects} Cell array with partition assignment vector
   %                   for each subject. Rows of partitionVec{subj} define
   %                   partition assignment of rows of Y{subj}.
   %                   Commonly these are the scanning run #s for beta
   %                   regressors.
   %                   If a single vector is provided, it is assumed to me the
   %                   same for all subjects 
   %
   %   conditionVec: {#Subjects} Cell array with condition assignment vector
   %                   for each subject. Rows of conditionVec{subj} define
   %                   condition assignment of rows of Y{subj}.
   %                   If a single vector is provided, it is assumed to me the
   %                   same for all subjects 
   %                   If the (elements of) conditionVec are matrices, it is
   %                   assumed to be the design matrix Z, allowing the
   %                   specification individualized models. 

Optional inputs are: 

.. sourcecode::python

   % OPTION:
   %   'runEffect': How to deal with effects that may be specific to different
   %                imaging runs:
   %                  'random': Models variance of the run effect for each subject
   %                            as a seperate random effects parameter.
   %                  'fixed': Consider run effect a fixed effect, will be removed
   %                            implicitly using ReML.
   %
   %   'isCheckDeriv: Check the derivative accuracy of theta params. Done using
   %                  'checkderiv'. This function compares input to finite
   %                  differences approximations. See function documentation.
   %
   %   'MaxIteration': Number of max minimization iterations. Default is 1000.
   %
   %   'S',S         : (Cell array of) NxN noise covariance matrices -
   %                   otherwise independence is assumed

And the outputs are defined as: 

.. sourcecode::python

   %   T:      Structure with following subfields:
   %       SN:                 Subject number
   %       likelihood:         log likelihood
   %       noise:              Noise parameter 
   %       run:                Run parameter (if run = 'random') 
   %       iterations:         Number of interations for model fit
   %       time:               Elapsed time in sec 
   %
   %   theta{m}     Cell array of estimated model parameters, each a 
   %                 #params x #numSubj matrix 
   %   G_pred{m}     Cell array of estimated G-matrices under the model 

The output can be used to compare the likelihoods between different models. Alternatively you can inspect the individual fits by looking at the parameters (theta) or the fitted second moment matrix (G_pred).


Example
^^^^^^^
In `pcm_recipe_feature`, we fit the feature model described above and then use the fitted parameters to determine the predicted correlation. 

.. sourcecode::python
   [D,theta,G_hat] = pcm_fitModelIndivid(Data,M,partVec,condVec,'runEffect','fixed');

   % Get the correlations from the parameters for Model1 
   var1        = theta{1}(1,:).^2;
   var2        = theta{1}(2,:).^2+theta{1}(3,:).^2;
   cov12       = theta{1}(1,:).*theta{1}(2,:);
   r_model1    = (cov12./sqrt(var1.*var2))'; 


Fitting to individual data sets with cross-validation across partitions 
-----------------------------------------------------------------------

Crossvalidation within subject ([@fig:Fig3]a) is the standard for encoding models and can also be applied to PCM-models.  Note that this part of the toolbox is still under development.

.. sourcecode::python

   function [T,D,theta_hat]=pcm_fitModelIndividCrossval(Y,M,partitionVec,conditionVec,varargin);

The use is very similar `pcm_fitModelIndivid`, except for two optional input parameters: 

Fitting to group data sets
--------------------------

The function `pcm_fitModelGroup` fits a model to a group of subjects. All parameters that change the **G** matrix, that is all `M.numGparams`, are shared across all subjects. To account for the individual signal-to-noise level, by default a separate signal strength ($s_i$) and noise parameter ($\sigma^{2}_{\epsilon,i}$) are fitted for each subject. That is, for each individual subject, the predicted covariance matrix of the data is:

.. math::
   {\bf{V}_i}=s_i\bf{ZGZ^{T}+S}\sigma^{2}_{\epsilon,i}.

To prevent scaling or noise variance parameters to become negative, we  actually optimise the log of these parameter, such that   

.. math::
   \begin{array}{c}
   s_i = exp(\theta_{s,i})\\
   \sigma^{2}_{\epsilon,i} = exp(\theta_{\epsilon, i}).
   \end{array}

The output `theta` for each model contains not only the `M.numGparams` model parameters, but also the noise parameters for all the subjects, then (optional) the scale parameters for all the subjects, and (if the runEffect is set to random)  the run-effect parameter for all the subjects.  The resultant scaling parameter for each subject is additionally stored under `T.scale` in the output structure. The fitting of an additional scale parameter can be switched off by providing the optional input argument `pcm_fitModelGroup(...,'fitScale',0)`.  Often, it speeds up the computation to perform a group fit first, so it can serve as a starting value for the crossvalidated group fit (see below). Otherwise the input and output parameters are identical to `pcm_fitModelIndivid`. 

Note that the introduction of the scale parameter introduces a certain amount of parameter redundancy. For example, if a model has only one single component and parameter, then the overall scaling is simply `s*theta`. One can remove the redundancy by forcing one model parameter to be 1 - in practice this, however, is not necessary.    

Fitting to group data sets with cross-validation across participants 
--------------------------------------------------------------------

PCM allows also between-subject crossvalidation ([@fig:Fig3]b). The model parameters that determine the representational structure are fitted to all the subjects together, using separate noise and scale parameters for each subject. Then the model is evaluated on the left-out subjects, after maximizing both scale and noise parameters. Function pcm_fitModelIndividCrossval.m implements these steps.

The function `pcm_fitModelGroupCrossval` implements these steps. As an additional input parameter, one can pass the parameters from the group fit as `(...,'groupFit',theta,...)`. Taking these as a starting point can speed up convergence.  

Example 
^^^^^^^

The function `pcm_recipe_finger` provides a full example how to use group crossvalidation to compare different models. Three models are being tested: A muscle model, a usage model (both a fixed models) and a combination model, in which both muscle and usage can be combined in any combination. Because the combination model has one more parameter than each single model, crossvalidation becomes necessary. Note that for the simple models, the simple group fit and the cross-validated group fit are identical, as in both cases only a scale and noise parameter are optimized for each subject. 

.. source::python

   % Model 1: Null model for baseline: here we use a model which has all finger 
   % Patterns be independent - i.e. all finger pairs are equally far away from
   % each other 
   M{1}.type       = 'component';
   M{1}.numGparams = 1;
   M{1}.Gc         = eye(5);
   M{1}.name       = 'null'; 

   % Model 2: Muscle model: derived from covariance structure of muscle
   % activity during single finger movements 
   M{2}.type       = 'component';
   M{2}.numGparams = 1;
   M{2}.Gc         = Model(1).G_cent;
   M{2}.name       = 'muscle'; 

   % Model 3: Natural statistics model: derived from covariance structure of
   % natual movements 
   M{3}.type       = 'component';
   M{3}.numGparams = 1;
   M{3}.Gc         = Model(2).G_cent;
   M{3}.name       = 'usage'; 

   % Model 4: Additive mixture between muscle and natural stats model
   M{4}.type       = 'component';
   M{4}.numGparams = 2;
   M{4}.Gc(:,:,1)  = Model(1).G_cent;
   M{4}.Gc(:,:,2)  = Model(2).G_cent;
   M{4}.name       = 'muscle + usage'; 

   % Model 5: Free model as Noise ceiling
   M{5}.type       = 'freechol'; 
   M{5}.numCond    = 5;
   M{5}.name       = 'noiseceiling'; 
   M{5}           = pcm_prepFreeModel(M{5}); 

   % Fit the models on the group level 
   [Tgroup,theta] = pcm_fitModelGroup(Y,M,partVec,condVec,'runEffect','fixed','fitScale',1);

   % Fit the models through cross-subject crossvalidation
   [Tcross,thetaCr] = pcm_fitModelGroupCrossval(Y,M,partVec,condVec,'runEffect','fixed','groupFit',theta,'fitScale',1);

   % Provide a plot of the crossvalidated likelihoods 
   subplot(2,3,[3 6]); 
   T = pcm_plotModelLikelihood(Tcross,M,'upperceil',Tgroup.likelihood(:,5)); 
   ```