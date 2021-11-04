Algorithms for deep motion generation
================================================================================

With the recent development of learning based methods, many works have focused on generative models for motion synthesis. Most learning-based methods treat joint motions as the prediction target while considering their spatial and temporal connections. In our library, we include a wide range of deep-learning methods from `RNN`, `VRNN`,`VAE`, `GAN` to `Transformer` that train generative models on different datasets:

.. toctree::
    learning/learning_rnn.rst
    learning/actor.rst
    learning/vae_lstm.rst
    learning/action2motion.rst