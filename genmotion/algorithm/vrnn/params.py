import torch
import os

from ..common.params import HyperParams

class HDM05Params(HyperParams):
    def __init__(self, mode="train") -> None:
        self.exp_mode = mode
        self.model_name = "vrnn"

        # if self.exp_mode == "train":
        self.learning_rate = 1e-4
        self.input_dim = 62
        self.output_dim = 62
        self.position_loss_weight = 0.1
        self.rotation_loss_weight = 1.0
        self.model_save_path = os.path.join(os.getcwd(),"pretrained_models", "vrnn4hdm05")
        self.frame_interval = 10
        self.input_motion_length = 50

        self.hidden_dim = 128
        self.n_layers = 1 # GRU layers
        self.z_dim = 32
        

        if self.exp_mode == "sample":
            self.max_target_length = 20
    