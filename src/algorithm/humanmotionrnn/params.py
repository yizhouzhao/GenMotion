############################ data loading ########################################
train_subject_ids = [1,6,7,8,9,11]  
""" training subject ids """
test_subject_ids = [5]  
"""test subject id"""
omit_one_hot = False
"""one-hot encoding when loading human3.6m dataset"""

############################ model ##############################################
architecture = "basic"
"""architecture version: Seq2seq architecture to use: [basic, tied]"""
HUMAN_SIZE = 54
"""human size"""
source_seq_len = 50
"""input sequence length"""
target_seq_len = 25
"""target sequence lenght"""

rnn_size = 1024
"""rnn hidden size"""
rnn_num_layers = 1
"""rnn layer num"""
max_gradient_norm = 5
"""maximum gradient norm"""

residual_velocities =  False
"""Add a residual connection that effectively models velocities"""


############################## Training ##################################
use_cuda = True # param: use_cuda?
batch_size = 16
"""batch size"""
learning_rate = 5e-3
"""learning rate"""
learning_rate_decay_factor = 0.95
"""learning rate decay"""
loss_to_use = "sampling_based"
"""The type of loss to use, supervised or sampling_based"""

############################## Logging ##################################
print_every = 50
"""printing frequency during training"""


