.. figure:: ../images/cover.png
   :alt: Title image
   :width: 100%

Welcome to GenMotion's documentation!
========================================
**GenMotion** (/gen’motion/) is a Python library for making skeletal animations. 
It enables easy dataset loading and experiment sharing for synthesizing skeleton-Based human animation with the Python API. It also comes with a easy-to-use and industry-compatible API for `Autodesk Maya <https://www.autodesk.com/products/maya/overview?term=1-YEAR&tab=subscription>`_,
`Maxon Cinema 4D <https://www.maxon.net/en/cinema-4d>`_, and `Blender <https://www.blender.org/>`_.


The source code is available on `Github <https://https://github.com/yizhouzhao/GenMotion>`_

.. note::

   This project is under active development.

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Getting Started

   installation

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: User Guide

   dataset
   model
   genmotion_api 
   genmotion_tutorials
   citation


Library overview
----------------

Working with datasets
^^^^^^^^^^^^^^^^^^^^^
We integrate multiple skeleton-based human motion datasets in GenMotion.
For datasets that have different parameterization of the body, we include 
documents for meta-data descriptions and visualization tools to illustrate characteristics of each dataset.

Benchmarking the state-of-the-arts
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
To encourage related research in human motion generation and retrieve empirical results from most advanced methods,
GenMotion re-produces the training procedure of character motion generation methods by reusing and cleaning the code from official implementation.

Rendering
^^^^^^^^^
To achieve real-time animation sampling, we provide communication interface, i.e. client and server interaction,  
with the 3D modeling software in GenMotion.