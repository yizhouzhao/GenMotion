# data loader for `On human motion prediction using recurrent neural networks`
# modified from original code https://github.com/enriccorona/human-motion-prediction-pytorch

import numpy as np
import torch
import copy

from .params import *

def readCSVasFloat(filename):
    """
  Borrowed from SRNN code. Reads a csv and returns a float matrix.
  https://github.com/asheshjain399/NeuralModels/blob/master/neuralmodels/utils.py#L34

  Args
    filename: string. Path to the csv file
  Returns
    returnArray: the read data in a float32 matrix
  """

    returnArray = []
    lines = open(filename).readlines()
    for line in lines:
        line = line.strip().split(',')
        if len(line) > 0:
            returnArray.append(np.array([np.float32(x) for x in line]))

    returnArray = np.array(returnArray)
    return returnArray

#!/usr/bin/python
# -*- coding: utf-8 -*-


def load_data(
    path_to_dataset,
    subjects,
    actions,
    one_hot,
    ):
    """
  Borrowed from SRNN code. This is how the SRNN code reads the provided .txt files
  https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/processdata.py#L270

  Args
    path_to_dataset: string. directory where the data resides
    subjects: list of numbers. The subjects to load
    actions: list of string. The actions to load
    one_hot: Whether to add a one-hot encoding to the data
  Returns
    trainData: dictionary with k:v
      k=(subject, action, subaction, 'even'), v=(nxd) un-normalized data
    completeData: nxd matrix with all the data. Used to normlization stats
  """

    nactions = len(actions)

    trainData = {}
    completeData = []
    for subj in subjects:
        for action_idx in np.arange(len(actions)):

            action = actions[action_idx]

            for subact in [1, 2]:  # subactions

                print ('Reading subject {0}, action {1}, subaction {2}'.format(subj,
                        action, subact))

                filename = \
                    '{0}/S{1}/{2}_{3}.txt'.format(path_to_dataset,
                        subj, action, subact)
                action_sequence = readCSVasFloat(filename)

                (n, d) = action_sequence.shape
                even_list = range(0, n, 2)

                if one_hot:

          # Add a one-hot encoding at the end of the representation

                    the_sequence = np.zeros((len(even_list), d
                            + nactions), dtype=float)
                    the_sequence[:, 0:d] = action_sequence[even_list, :]
                    the_sequence[:, d + action_idx] = 1
                    trainData[(subj, action, subact, 'even')] = \
                        the_sequence
                else:
                    trainData[(subj, action, subact, 'even')] = \
                        action_sequence[even_list, :]

                if len(completeData) == 0:
                    completeData = copy.deepcopy(action_sequence)
                else:
                    completeData = np.append(completeData,
                            action_sequence, axis=0)

    return (trainData, completeData)



def normalize_data( data, data_mean, data_std, dim_to_use, actions, one_hot ):
    """
    Normalize input data by removing unused dimensions, subtracting the mean and
    dividing by the standard deviation

    Args
    data: nx99 matrix with data to normalize
    data_mean: vector of mean used to normalize the data
    data_std: vector of standard deviation used to normalize the data
    dim_to_use: vector with dimensions used by the model
    actions: list of strings with the encoded actions
    one_hot: whether the data comes with one-hot encoding
    Returns
    data_out: the passed data matrix, but normalized
    """
    data_out = {}
    nactions = len(actions)

    if not one_hot:
        # No one-hot encoding... no need to do anything special
        for key in data.keys():
            data_out[ key ] = np.divide( (data[key] - data_mean), data_std )
            data_out[ key ] = data_out[ key ][ :, dim_to_use ]

    else:
        # TODO hard-coding 99 dimensions for un-normalized human poses
        for key in data.keys():
            data_out[ key ] = np.divide( (data[key][:, 0:99] - data_mean), data_std )
            data_out[ key ] = data_out[ key ][ :, dim_to_use ]
            data_out[ key ] = np.hstack( (data_out[key], data[key][:,-nactions:]) )

    return data_out

def normalization_stats(completeData):
    """"
    Also borrowed for SRNN code. Computes mean, stdev and dimensions to ignore.
    https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/processdata.py#L33

    Args
    completeData: nx99 matrix with data to normalize
    Returns
    data_mean: vector of mean used to normalize the data
    data_std: vector of standard deviation used to normalize the data
    dimensions_to_ignore: vector with dimensions not used by the model
    dimensions_to_use: vector with dimensions used by the model
    """
    data_mean = np.mean(completeData, axis=0)
    data_std  =  np.std(completeData, axis=0)

    dimensions_to_ignore = []
    dimensions_to_use    = []

    dimensions_to_ignore.extend( list(np.where(data_std < 1e-4)[0]) )
    dimensions_to_use.extend( list(np.where(data_std >= 1e-4)[0]) )

    data_std[dimensions_to_ignore] = 1.0

    return data_mean, data_std, dimensions_to_ignore, dimensions_to_use


class HumanMotionDataset(torch.utils.data.Dataset):
    
    # define normalization statistics
    data_mean = None
    data_std = None
    dim_to_ignore = None
    dim_to_use = None

    def __init__(self, data_dir, subject_ids, actions, one_hot, is_train = True):
            super().__init__()  
            # load data
            self.data_dir = data_dir
            self.subject_ids = subject_ids
            self.actions = actions
            self.one_hot = one_hot
            self.raw_data, self.complete_dataset = load_data( data_dir, subject_ids, actions, one_hot)
        
            # model_input information
            self.input_size = HUMAN_SIZE + len(self.actions)
            
            # compute normalization stats and normalize data
            self.is_train = is_train # is training dataset or not
            if self.is_train:
                HumanMotionDataset.data_mean, HumanMotionDataset.data_std, HumanMotionDataset.dim_to_ignore, HumanMotionDataset.dim_to_use = normalization_stats(self.complete_dataset)
                self.normalized_data = normalize_data( self.raw_data, self.data_mean, self.data_std, self.dim_to_use, self.actions, self.one_hot )
            else:
                assert HumanMotionDataset.data_mean is not None, "Must initialize training dataset first"
                self.normalized_data = normalize_data( self.raw_data, HumanMotionDataset.data_mean, HumanMotionDataset.data_std, HumanMotionDataset.dim_to_use, self.actions, self.one_hot )
      
            # calculate dataset length
            self.all_keys = list(self.normalized_data)
            self.length = self._calculate_data_length()

    def _calculate_data_length(self, fix_length = 10000):
        # calculate the total length of the dataset
        if fix_length > 0:
            if self.is_train:
                return fix_length
            else:
                return fix_length // 10

        # TODO: full-dataset training, which requires a lot of efforts.
        # else:
    
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        chosen_key = np.random.choice(len(self.all_keys))

        # How many frames in total do we need?
        total_frames = source_seq_len + target_seq_len
        the_key = self.all_keys[chosen_key]
        
        # Get the number of frames
        n, _ = self.normalized_data[ the_key ].shape

        # Sample somewherein the middle
        idx = np.random.randint( 16, n-total_frames )

        # Select the data around the sampled points
        data_sel = self.normalized_data[ the_key ][idx:idx+total_frames ,:]


        # Add the data
        encoder_input = data_sel[0:source_seq_len-1, :]
        decoder_input = data_sel[source_seq_len-1:source_seq_len + target_seq_len-1, :]
        decoder_output = data_sel[source_seq_len:, 0:self.input_size]

        return encoder_input, decoder_input, decoder_output




