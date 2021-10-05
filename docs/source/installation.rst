Installation
=============


To use GenMotion, first download the source code

.. code-block:: console

   (.venv) $ git clone https://github.com/yizhouzhao/GenMotion.git

or directly download from `github <https://https://github.com/yizhouzhao/GenMotion>`_

Configuration
--------------

To retrieve a list of random ingredients,
you can use the ``lumache.get_random_ingredients()`` function:


The ``kind`` parameter should be either ``"meat"``, ``"fish"``,
or ``"veggies"``. Otherwise, :py:func:`lumache.get_random_ingredients`
will raise an exception.


