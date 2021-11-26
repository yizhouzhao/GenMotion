*********
GenMotion
*********

.. image:: https://github.com/yizhouzhao/genmotion/actions/workflows/CI.yml/badge.svg?branch=main
   :target: https://github.com/yizhouzhao/genmotion/actions/workflows/CI.yml
   :alt: CI

.. image:: https://readthedocs.org/projects/genmotion/badge/?version=latest
   :target: https://genmotion.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status

.. image:: https://img.shields.io/pypi/v/genmotion
   :target: https://genmotion.readthedocs.io/en/latest/?badge=latest
   :alt: PyPI
   
.. image:: https://img.shields.io/github/license/yizhouzhao/genmotion
   :target: https://choosealicense.com/licenses/mit/
   :alt: Licence
   
`GenMotion` (/gen’motion/) is a Python library for making skeletal animations. 
GenMotion provide unified pipelines for loading data, diversified motion generation models, and real-time animation sampling. It also comes with a easy-to-use and industry-compatible API for `Autodesk Maya <https://www.autodesk.com/products/maya/overview?term=1-YEAR&tab=subscription>`_,
`Maxon Cinema 4D <https://www.maxon.net/en/cinema-4d>`_, and `Blender <https://www.blender.org/>`_.

You can find the full ducumentation and tutorials `here <https://genmotion.readthedocs.io/en/latest/>`_.

.. contents:: **Contents of this document:**
   :depth: 2


Progress
========

Rendering Tools x Datasets
--------------------------

test

+---------------+---------------+---------------+---------------+
|               | Maya          | C4D           | Blender       |
+===============+===============+===============+===============+
| HDM05         | [Done] 10/06  |               |               |
+---------------+---------------+---------------+---------------+
| Mocap         | [Done] 10/07  |               |               |
+---------------+---------------+---------------+---------------+
| Human3.6m     |               |               |               |
+---------------+---------------+---------------+---------------+
| Social        |               |               |               |
+---------------+---------------+---------------+---------------+
| NTU rgbd      |               |               |               |
+---------------+---------------+---------------+---------------+
| AMASS         | [Done] 10/08  |               | [Done] 10/14  |
+---------------+---------------+---------------+---------------+
| Mixamo        | [Done]        |               | [Done]        |
+---------------+---------------+---------------+---------------+

Model x Dataset
---------------

+--------------------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+
|                    | HDM05     | Mocap     | Human3.6m | Social    | TU rgbd   | AMASS     | Mixamo    | 
+--------------------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+
|Motion VAE          |           |           |           |           |           |           |           |
+--------------------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+
|Motion Transformer  |           |           |           |           |           |           |           |
+--------------------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+
|Motion RNN          |           |           |           |           |           |           |           |
+--------------------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+
|Motion VRNN         |           |           |           |           |           |           |           |
+--------------------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+
|MoGlow              |           |           |           |           |           |           |           |
+--------------------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+
|MT-VAE              |           |           |           |           |           |           |           |
+--------------------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+


Installation
============

(This is still under construction)

You can install ``GenMotion`` directly from the pip library with:

.. code:: shell

    pip install genmotion
