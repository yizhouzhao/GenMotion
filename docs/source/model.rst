Model overview
==========================================

Recently, Deep Neural Networks (DNNs) have shown their power to improve the performance of character animation, as evidenced by the growing number of publications on the topic of deep motion synthesis. In animation, deep-learning based approaches attempt to handle the complicated human motion and provide promising perspectives for cheaper and faster
animation making. We list several methods for skeletal animation in  `GenMotion`, and we are continuing building and expanding methods for deep motion synthesis in the future release of `GenMotion`.

Sequence to Sequence (Seq2Seq)
################################################################

Seq2Seq models are popular and widely used in motion prediction/ Seq2Seq-based approaches generally consist in training a
Human Motion Recurrent Neural Network (RNN) as encoder to map input to a hidden vector, and training another RNN as decoder to generate motion from the hidden vector :cite:`julieta2017motion`. Both the encdoer and decoder are trained jointly.

Recurrent Network Models for Human Dynamics 
################################################################

Encoder-Recurrent-Decoder (ERD) :cite:`fragkiadaki2015recurrent`. is a model for prediction of human body poses from motion capture. The ERD model is a recurrent neural network that incorporates nonlinear encoder and decoder networks before and after recurrent layers.

Variational Autoencoder
################################################################

Variational autoencoders (VAEs) are a deep learning technique for learning latent representations. They have also been used to draw images, achieve state-of-the-art results in semi-supervised learning, as well as interpolate between sentences. 

VAEs can also be applied into generation human motion by combining samples from a variational approximation to the intractable posterior :cite:`habibie2017recurrent`, or be integrated into sequence generation tasks from a Variational Recurrent Neural Network (VRNN) :cite:`chung2015recurrent`.

Transformer
################################################################


.. bibliography:: refs.bib