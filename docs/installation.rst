Installation
============

Dependencies
------------

The required dependencies to use the software are:

* python >= 3.6,
* numpy >= 1.16
* pandas >= 0.24
* matplotlib >= 1.5.1
* seaborn

If you want to run the tests, you need pytest >= 3.9 and pytest-cov for coverage reporting.


Installation using Pip
----------------------

PcmPy is available on PyPi with::

pip install PcmPy

Installation for developers
---------------------------

You can also clone or fork the whole repository from https://github.com/diedrichsenlab/PCMPy. Place the the entire repository in a folder of your choice. Then add the folder by adding the following lines to your ``.bash.profile`` or other shell startup file::

    PYTHONPATH=/DIR/PcmPy:${PYTHONPATH}
    export PYTHONPATH

You then should be able to import the entire toolbox as::

    import PcmPy as pcm






