# Pattern Component Modelling toolbox (Python) 

Pattern component modeling (PCM) is a practical Bayesian approach for evaluating representational models - models that specify how complex patterns of neural activity relate to visual stimuli, motor actions, or abstract thoughts. Similar to encoding models, PCM evaluates the ability of models to predict novel brain activity patterns. In contrast to encoding models, however, the activity of individual voxels across conditions (activity profiles) is not directly fitted. Rather, PCM integrates over all possible activity profiles and computes the marginal likelihood of the data under the activity profile distribution specified by the representational model. By using an analytical expression for the marginal likelihood, PCM allows the fitting of flexible representational models, in which the relative strength and form of different feature sets can be estimated from the data. 

For more background::
* Diedrichsen, J. (2018). Representational models and the feature fallacy. In M. S. Gazzaniga (Ed.), The Cognitive Neurosciences. 
* Diedrichsen, J., Yokoi, A., & Arbuckle, S. A. (2018). Pattern component modeling: A flexible approach for understanding the representational structure of brain activity patterns. Neuroimage. 180(Pt A), 119-133.
* Diedrichsen, J., & Kriegeskorte, N. (2017). Representational models: A common framework for understanding encoding, pattern-component, and representational-similarity analysis. PLoS Comput Biol. 
* Diedrichsen, J., Ridgway, G., Friston, K.J., Wiestler, T., (2011). Comparing the similarity and spatial structure of neural representations: A pattern-component model. Neuroimage.


Full documentation can be found on:
[https://pcm-toolbox-python.readthedocs.io]

For a verions of this toolbox in Matlab, For a version of the toolbox in Python, see [https://github.com/jdiedrichsen/pcm_toolbox].

