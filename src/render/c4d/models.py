#!/usr/bin/python
# -*- coding: utf-8 -*-

"""Sequence-to-sequence model for human motion prediction."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .params import use_cuda

class Seq2SeqModel(nn.Module):
    """Sequence-to-sequence model for human motion prediction"""

    def __init__(
        self,
        architecture,
        source_seq_len,
        target_seq_len,
        rnn_size,
        num_layers,
        max_gradient_norm,
        batch_size,
        learning_rate,
        learning_rate_decay_factor,
        loss_to_use,
        number_of_actions,
        one_hot=True,
        residual_velocities=False,
        dropout=0.0,
        dtype=torch.float32,
        ):
        """Create the model.

    Args:
      architecture: [basic, tied] whether to tie the decoder and decoder.
      source_seq_len: lenght of the input sequence.
      target_seq_len: lenght of the target sequence.
      rnn_size: number of units in the rnn.
      num_layers: number of rnns to stack.
      max_gradient_norm: gradients will be clipped to maximally this norm.
      batch_size: the size of the batches used during training;
        the model construction is independent of batch_size, so it can be
        changed after initialization if this is convenient, e.g., for decoding.
      learning_rate: learning rate to start with.
      learning_rate_decay_factor: decay learning rate by this much when needed.
      loss_to_use: [supervised, sampling_based]. Whether to use ground truth in
        each timestep to compute the loss after decoding, or to feed back the
        prediction from the previous time-step.
      number_of_actions: number of classes we have.
      one_hot: whether to use one_hot encoding during train/test (sup models).
      residual_velocities: whether to use a residual connection that models velocities.
      dtype: the data type to use to store internal variables.
    """

                         # hidden recurrent layer size

        super(Seq2SeqModel, self).__init__()

        self.HUMAN_SIZE = 54
        self.input_size = (self.HUMAN_SIZE
                           + number_of_actions if one_hot else self.HUMAN_SIZE)

        print('One hot is ', one_hot)
        print('Input size is %d' % self.input_size)

    # Summary writers for train and test runs

        self.source_seq_len = source_seq_len
        self.target_seq_len = target_seq_len
        self.rnn_size = rnn_size
        self.batch_size = batch_size
        self.dropout = dropout

    # === Create the RNN that will keep the state ===

        print('rnn_size = {0}'.format(rnn_size))
        self.cell = torch.nn.GRUCell(self.input_size, self.rnn_size)

#    self.cell2 = torch.nn.GRUCell(self.rnn_size, self.rnn_size)

        self.fc1 = nn.Linear(self.rnn_size, self.input_size)

    def forward(self, encoder_inputs, decoder_inputs):

        def loop_function(prev, i):
            return prev

        batchsize = encoder_inputs.shape[0]
        encoder_inputs = torch.transpose(encoder_inputs, 0, 1)
        decoder_inputs = torch.transpose(decoder_inputs, 0, 1)

        state = torch.zeros(batchsize, self.rnn_size)
        #state2 = torch.zeros(batchsize, self.rnn_size)

        if use_cuda:
            state = state.cuda()
            #state2 = state2.cuda()

        for i in range(self.source_seq_len - 1):
            state = self.cell(encoder_inputs[i], state)
            #state2 = self.cell2(state, state2)

            state = F.dropout(state, self.dropout,
                              training=self.training)
            if use_cuda:
                state = state.cuda()
                #state2 = state2.cuda()

        outputs = []
        prev = None
        for (i, inp) in enumerate(decoder_inputs):
            if loop_function is not None and prev is not None:
                inp = loop_function(prev, i)

            inp = inp.detach()

            state = self.cell(inp, state)

            #      state2 = self.cell2(state, state2)
            #      output = inp + self.fc1(state2)
            #      state = F.dropout(state, self.dropout, training=self.training)`

            output = inp + self.fc1(F.dropout(state, self.dropout,
                                    training=self.training))

            outputs.append(output.view([1, batchsize, self.input_size]))
            if loop_function is not None:
                prev = output

            #return outputs, state

        outputs = torch.cat(outputs, 0)
        return torch.transpose(outputs, 0, 1)
    
    def sample(self, encoder_inputs):

        def loop_function(prev, i):
            return prev
        
        batchsize, length, _ = encoder_inputs.shape

        encoder_inputs, inp = encoder_inputs[:,:length - 1,:], encoder_inputs[:,length - 1,:]
        encoder_inputs = torch.transpose(encoder_inputs, 0, 1)

        state = torch.zeros(batchsize, self.rnn_size)

        if use_cuda:
            state = state.cuda()
        

        for i in range(length - 1):
            state = self.cell(encoder_inputs[i], state)
            state = F.dropout(state, self.dropout,
                              training=self.training)
            if use_cuda:
                state = state.cuda()
        
        outputs = []
        prev = None
        for i in range(self.target_seq_len):
            
            if loop_function is not None and prev is not None:
                inp = loop_function(prev, i)

            inp = inp.detach()

            state = self.cell(inp, state)

            #      state2 = self.cell2(state, state2)
            #      output = inp + self.fc1(state2)
            #      state = F.dropout(state, self.dropout, training=self.training)`

            output = inp + self.fc1(F.dropout(state, self.dropout,
                                    training=self.training))

            outputs.append(output.view([1, batchsize, self.input_size]))
            if loop_function is not None:
                prev = output

            #return outputs, state

        outputs = torch.cat(outputs, 0)
        return torch.transpose(outputs, 0, 1)