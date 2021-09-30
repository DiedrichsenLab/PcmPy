.. PCM_toolbox index

Pattern Component Modelling (PCM) toolbox
=========================================
The Pattern Component Modelling (PCM) toolbox is designed to analyze multivariate brain activity patterns in a Bayesian approach. The theory is laid out in Diedrichsen et al. (2017) as well as in this documentation. We provide details for model specification, model estimation, visualisation, and model comparison. The documentation also refers to the empricial examples in demos folder.

The original Matlab version of the toolbox is available at https://github.com/jdiedrichsen/pcm_toolbox. The practical examples in this documentation is for the Python, which can be found at https://github.com/diedrichsenlab/PCMPy
Note that the toolbox does not provide functions to extract the required data from the first-level GLM or raw data, or to run search-light or ROI analyses. We have omitted these function as they strongly depend on the analysis package used for the basic imaging analysis. Some useful tools for the extraction of multivariate data from  first-level GLMs can be found in the RSA-toolbox (https://github.com/rsagroup/rsatoolbox) and of course nilearn.

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
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
