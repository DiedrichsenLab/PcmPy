.. PCM_toolbox index

Pattern Component Modelling (PCM) toolbox
=========================================
The Pattern Component Modelling (PCM) toolbox is designed to analyze multivariate brain activity patterns in a Bayesian approach. The theory is laid out in Diedrichsen et al. (2017) as well as in this documentation. We provide details for model specification, model estimation, visualisation, and model comparison. The documentation also refers to the empricial examples in demos folder.

The original Matlab version of the toolbox is available at https://github.com/jdiedrichsen/pcm_toolbox. The practical examples in this documentation are available as jupyter notebooks at https://github.com/DiedrichsenLab/PcmPy/tree/master/docs/demos. 

Note that the toolbox does not provide functions to extract the required data from the first-level GLM or raw data, or to run search-light or ROI analyses. We have omitted these function as they strongly depend on the analysis package used for the basic imaging analysis. Some useful tools for the extraction of multivariate data from  first-level GLMs can be found in the RSA-toolbox (https://github.com/rsagroup/rsatoolbox) and of course nilearn.


Licence and Acknowledgements
----------------------------
The PCMPy toolbox is being developed by members of the Diedrichsenlab including Jörn Diedrichsen, Giacomo Ariani, Spencer Arbuckle, Eva Berlot, and Atsushi Yokoi. It is distributed under MIT License, meaning that it can be freely used and re-used, as long as proper attribution in form of acknowledgments and links (for online use) or citations (in publications) are given. When using, please cite the relevant references:

* Diedrichsen, J., Yokoi, A., & Arbuckle, S. A. (2018). Pattern component modeling: A flexible approach for understanding the representational structure of brain activity patterns. Neuroimage. 180(Pt A), 119-133.
* Diedrichsen, J., Ridgway, G., Friston, K.J., Wiestler, T., (2011). Comparing the similarity and spatial structure of neural representations: A pattern-component model. Neuroimage.


Documentation
-------------
.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation.rst
   introduction.rst
   model.rst
   fitting.rst
   visualisation.rst
   inference.rst
   regression.rst
   examples.rst
   math.rst
   reference.rst
   literature_cited.rst

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
