.. PCM_toolbox index

Pattern Component Modelling (PCM) toolbox
=========================================
This manual provides an introduction to how to use the Pattern Component Modelling (PCM) toolbox. The theory behind this approach is laid out in an accompanying paper (Diedrichsen et al., 2017) - but the main ideas are described in the introduction. We then provide more details for model specification, model estimation, visualisation, and model comparison. The documentation also refers to the empricial examples in demos folder.

Note that the toolbox does not provide functions to extract the required data from the first-level GLM or raw data, or to run search-light or ROI analyses. We have omitted these function as they strongly depend on the analysis package used for the basic imaging analysis. Some useful tools for the extraction of multivariate data from  first-level GLMs can be found in the [RSA-toolbox](https://github.com/rsagroup/rsatoolbox) and [Surfing toolbox](https://github.com/nno/surfing).

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   introduction.rst
   model.rst
   fitting.rst
   visualisation.rst
   inference.rst
   regression.rst
   math.rst
   reference.rst
   literature_cited.rst

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
