# Pattern Component Modelling toolbox (Python)

Pattern component modeling (PCM) is a likelihood approach for evaluating representational models - models that specify how complex patterns of neural activity relate to visual stimuli, motor actions, or abstract thoughts. Similar to encoding models, PCM evaluates the ability of models to predict novel brain activity patterns. In contrast to encoding models, however, the activity of individual voxels across conditions (activity profiles) is not directly fitted. Rather, PCM integrates over all possible activity profiles and computes the marginal likelihood of the data under the activity profile distribution specified by the representational model. By using an analytical expression for the marginal likelihood, PCM allows the fitting of flexible representational models, in which the relative strength and form of different feature sets can be estimated from the data.

This is a repository for the Python version. For a MATLAB verion of this toolbox see [here](https://github.com/jdiedrichsen/pcm_toolbox).

### Documentation
Full documentation can be found [here](https://pcm-toolbox-python.readthedocs.io)

### Licence and Acknowledgements
The PCMPy toolbox is being developed by members of the Diedrichsenlab including Jörn Diedrichsen, Giacomo Ariani, Spencer Arbuckle, Eva Berlot, and Atsushi Yokoi. It is distributed under MIT License, meaning that it can be freely used and re-used, as long as proper attribution in form of acknowledgments and links (for online use) or citations (in publications) are given. The relevant references are:

* Diedrichsen, J., Yokoi, A., & Arbuckle, S. A. (2018). Pattern component modeling: A flexible approach for understanding the representational structure of brain activity patterns. Neuroimage. 180(Pt A), 119-133. [[link]](https://www.diedrichsenlab.org/pubs/Neuroimage_2017.pdf)
* Diedrichsen, J., Ridgway, G., Friston, K.J., Wiestler, T., (2011). Comparing the similarity and spatial structure of neural representations: A pattern-component model. Neuroimage. [[link]](https://www.diedrichsenlab.org/pubs/Neuroimage_pattern_2011.pdf)

For more theoretical background: 

* Diedrichsen, J. (2018). Representational models and the feature fallacy. In M. S. Gazzaniga (Ed.), The Cognitive Neurosciences. [[link]](https://www.diedrichsenlab.org/pubs/RepresentationalModels_2018.pdf)
* Diedrichsen, J., & Kriegeskorte, N. (2017). Representational models: A common framework for understanding encoding, pattern-component, and representational-similarity analysis. PLoS Comput Biol. [[link]](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005508)
