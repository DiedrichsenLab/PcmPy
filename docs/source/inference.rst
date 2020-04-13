.. _inference:

Inference
=========

Inference on model parameters
-----------------------------

First we may make inferences based on the parameters of a single fitted model. The parameter may be the weight of a specific component or another metric derived from the second moment matrix. For example, the estimated correlation coefficient between condition 1 and 2 would be :math:`r_{1,2}=\mathbf{G}_{1,2}/\sqrt{\mathbf{G}_{1,1}\mathbf{G}_{2,2}}`. We may want to test whether the correlation between the patterns is larger than zero, or whether a parameter differs between two different subject groups, two different regions, or whether they change with experimental treatments.

The simplest way of testing parameters would be to use the point estimates from the model fit from each subject and apply frequentist statistics to test different hypotheses, for example using a t- or F-test. Alternatively, one can obtain estimates of the posterior distribution of the parameters using MCMC sampling [@RN3567] or Laplace approximation [@RN3255]. This allows the application of Bayesian inference, such as the report of credibility intervals. 

One important limitation to keep in mind is that parameter estimates from PCM are not unbiased in small samples. This is caused because estimates of :math:`\mathbf{G}` are constrained to be positive definite. This means that the variance of each feature must be larger or equal to zero. Thus, if we want to determine whether a single activity pattern is different from baseline activity, we cannot simply test our variance estimate (i.e. elements of :math:`\mathbf{G}`) against zero - they trivially will always be larger, even if the true variance is zero. Similarly, another important statistic that measures the pattern separability or classifiability of two activity patterns, is the Euclidean distance, which can be calculated from the second moment matrix as :math:`d=\mathbf{G}_{1,1}+\mathbf{G}_{2,2}-2\mathbf{G}_{1,2}`. Again, given that our estimate of :math:`\mathbf{G}` is positive definite, any distance estimate is constrained to be positive. To determine whether two activity patterns are reliably different, we cannot simply test these distances against zero, as the test will be trivially larger than zero. A better solution for inferences from individual parameter estimates is therefore to use a cross-validated estimate of the second moment matrix  and the associated distances [@RN3565][@RN3543]. In this case the expected value of the distances will be zero, if the true value is zero. As a consequence, variance and distance estimates can become negative. These techniques, however, take us out of the domain of PCM and into the domain of representational similarity analysis [@RN2697][@RN3672]. 

Inference on model evidence 
---------------------------

As an alternative to parameter-based inference, we can fit multiple models and compare them according to their model evidence; the likelihood of the data given the models (integrated over all parameters). In encoding models, the weights :math:`\mathbf{W}` are directly fitted to the data, and hence it is important to use cross-validation to compare models with different numbers of features. The marginal likelihood already integrates all over all likely values of :math:`\mathbf{U}`, and hence :math:`\mathbf{W}`, thereby removing the bulk of free parameters. Thus, in practice the marginal likelihood will be already close to the true model evidence.

Our marginal likelihood, however, still depends on the free parameters :math:`\boldsymbol{\theta}`. So, when comparing models, we need to still account for the risk of overfitting the model to the data. For fixed models, there are only two free parameters: one relating to the strength of the noise (:math:`\theta_{\epsilon}`) and one relating to the strength of the signal (:math:`\theta_s`). This compares very favorably to the vast number of free parameters one would have in an encoding model, which is the size of :math:`\mathbf{W}`, the number of features x number of voxels. However, even the fewer model parameters still need to be accounted for. We consider here four ways of doing so.

The first option is to use empirical Bayes or Type-II maximal likelihood. This means that we simply replace the unknown parameters with the point estimates that maximize the marginal likelihood. This is in general a feasible strategy if the number of free parameters is low and all models have the same numbers of free parameters, which is for example the case when we are comparing different fixed models. The two free parameters here determine the signal-to-noise ratio. For models with different numbers of parameters we can penalize the likelihood by :math:`\frac{1}{2}d_{\theta}\log(n)` , yielding the Bayes information criterion (BIC) as the approximation to model evidence.

As an alternative option, we can use cross-validation within the individual (\hyperref[fig2]{Fig. 2a}) to prevent overfitting for more complex flexible models, as is also currently common practice for encoding models [@RN3096]. Taking one imaging run of the data as test set, we can fit the parameters to data from the remaining runs. We then evaluate the likelihood of the left-out run under the distribution specified by the estimated parameters. By using each imaging run as a test set in turn, and adding the log-likelihoods (assuming independence across runs), we thus can obtain an approximation to the model evidence. Note, however, that for a single (fixed) encoding model, cross-validation is not necessary under PCM, as the activation parameters for each voxel (:math:`\mathbf{W}` or :math:`\mathbf{U}`) are integrated out in the likelihood. Therefore, it can be handled with the first option we described above. 

For the third option, if we want to test the hypothesis that the representational structure in the same region is similar across subjects, we can perform cross-validation across participants (\hyperref[fig2]{Fig. 2b}). We can estimate the parameters that determine the representational structure using the data from all but one participant and then evaluate the likelihood of data from the left-out subject under this distribution. When performing cross-validation within individuals, a flexible model can fit the representational structure of individual subjects in different ways, making the results hard to interpret. When using the group cross-validation strategy, the model can only fit a structure that is common across participants. Different from encoding models, representational models can be generalized across participants, as we do not fit the actual activity patterns, but rather the representational structure. In a sense, this method is performing  "hyper alignment''  [@RN3572] without explicitly calculating the exact mapping into voxel space. When using this approach, we still allow each participant to have its own signal and noise parameters, because the signal-to-noise ratio is idiosyncratic to each participant's data. When evaluating the likelihood of left-out data under the estimated model parameters, we therefore plug in the ML-estimates for these two parameters for each subject. 

Finally, a last option is to implement a full Bayesian approach and to impose priors on all parameters, and then use a Laplace approximation to estimate the model evidence[@RN3654][@RN3255]. While it certainly can be argued that this is the most elegant approach, we find that cross-validation at the level of model parameters provides us with a practical, straightforward, and transparent way of achieving a good approximation.

Each of the inference strategies supplies us with an estimate of the model evidence. To compare models, we then calculate the log Bayes factor, which is the difference between the log model evidences.

.. math::
   \begin{array}{c}
   \log B_{12} = \log \frac{p(\mathbf{Y}|M_1)}{p(\mathbf{Y}|M_2)}\\
   =\log p(\mathbf{Y}|M_1)-\log p(\mathbf{Y}|M_2)
   \end{array}

Log Bayes factors of over 1 are usually considered positive evidence and above 3 strong evidence for one model over the other [@RN3654].

Group inference
---------------

How to perform group inference in the context of Bayesian model comparison is a topic of ongoing debate in the context of neuroimaging. A simple approach is to assume that the data of each subject is independent (a very reasonable assumption) and that the true model is the same for each subject (a maybe less reasonable assumption). This motivates the use of log Group Bayes Factors (GBF), which is simple sum of all individual log Bayes factor across all subjects n

.. math::
   \log GBF = \sum_{n} log B_{n}.


Performing inference on the GBF is basically equivalent to a fixed-effects analysis in neuroimaging, in which we combine all time series across subjects into a single data set, assuming they all were generated by the same underlying model. A large GBF therefore could be potentially driven by one or few outliers. We believe that the GBF therefore does not provide a desirable way of inferring on representational models - even though it has been widely used in the comparison of DCM models [@RN2029].

At least the distribution of individual log Bayes factors should be reported for each model. When evaluating model evidences against a Bayesian criterion, it can be useful to use the average log Bayes factor, rather than the sum. This stricter criterion is independent of sample size, and therefore provides a useful estimate or effect size. It expresses how much the favored model is expected to perform better on a new, unseen subject. We can also use the individual log Bayes factors as independent observations that are then submitted to a frequentist test, using either a t-, F-, or nonparametric test. This provides a simple, practical approach that we will use in our examples here. Note, however, that in the context of group cross-validation, the log-Bayes factors across participants are not strictly independent. 

Finally, it is also possible to build a full Bayesian model on the group level, assuming that the winning model is different for each subject and comes from a multinomial distribution with unknown parameters [@RN3653].

Noise ceilings
--------------

Showing that a model provides a better explanation of the data as compared to a simpler Null-model is an important step. Equally important, however, is to determine how much of the data the model does not explain. Noise ceilings[@RN3300] provide us with an estimate of how much systematic structure (either within or across participants) is present in the data, and what proportion is truly random. In the context of PCM, this can be achieved by fitting a fully flexible model, i.e. a free model in which the second moment matrix can take any form. The non-cross-validated fit of this model provides an absolute upper bound - no simpler model will achieve a higher average likelihood. As this estimate is clearly inflated (as it does not account for the parameter fit) we can also evaluate the free model using cross-validation. Importantly, we need to employ the same cross-validation strategy (within \slash between subjects) as used with the models of interest. If the free model performs better than our model of interest even when cross-validated, then we know that there are definitely aspects of the representational structure that the model did not capture. If the free model performs worse, it is overfitting the data, and our currently best model provides a more concise description of the data. In this sense, the performance of the free model in the cross-validated setting provides a  lower bound to the noise ceiling. It still may be the case that there is a better model that will beat the currently best model, but at least the current model already provides an adequate description of the data. Because they are so useful, noise ceilings should become a standard reporting requirement when fitting representational models to fMRI data, as they are in other fields of neuroscientific inquiry already. The Null-model and the upper noise ceiling also allow us to normalize the log model evidence to be between 0 (Null-model) and 1 (noise ceiling), effectively obtaining a Pseudo-:math:`R^{2}`. 

Inference on model components
-----------------------------

Often, we want to test which model components are required to explain the observed activity patterns. For example, for sequence representations, we may consider as  model components the representation of single fingers, finger transitions, or whole sequences [@RN3713]. To assess the importance of each of the components, we could fit each components seperately and test how much the marginal likelihood increases relative to the Null-model (*knock-in*). We can also fit the full model containing all components and then assess how much the marginal likelihood decreases when we leave a single model component out (*knock-out*). The most comprehensive approach, however, is to fit all combinations of components separately [@RN3717]. 

To do this, we can construct a model family containing all possible combination models by switching the individual components either on or off. If we have :math:`k` components of interest, we will end up with :math:`2^k` models. The following  pcm toolbox function constructs such a 'model family' given all the individual components (`MComp`)  

`[M,CompI] = pcm_constructModelFamily(MComp);`

`M` is the resultant model family containing :math:`2^k` models and `CompI` is an indicator matrix of size :math:`2^k k` denoting for each of :math:`k` components whether it is included or excluded in each model. 

After fitting all possible model combinations, one could simply select the model combination with the highest marginal likelihood. A more stable and informative approach, however, is to determine the posterior likelihood for each model component, averaged across all possible model combinations [@RN3716]. In the context of a model family, we can calculate the posterior probability of a model component being present (:math:`F=1`) from the summed posteriors of all  models that contained that component (:math:`M:F=1`)

.. math::
   p(F=1|data)= \frac{\sum_{M:F=1}{p(data|M) p(M)}}{\sum_{M}{p(data|M)p(M)}}

Finally, we can also obtain a Bayes factor as a measure of the evidence that the component is present

.. math::
   BF_{F=1}= \frac{\sum_{M:F=1}{p(data|M) }}{\sum_{M:F=0}{p(data|M)}}

After fitting a complete model family, you can calculate the above quantities using the function  `pcm_componentPosterior`. The Bayes factor for each model component can then be interpreted and tested on the group level, as outlined in the section on group inference. 

