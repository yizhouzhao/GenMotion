import torch
import os

class HDM05Params:
    def __init__(self, mode="train") -> None:
        self.exp_mode = mode

        # if self.exp_mode == "train":
        self.learning_rate = 1e-4
        self.input_dim = 62
        self.output_dim = 62
        self.position_loss_weight = 0.1
        self.rotation_loss_weight = 1.0
        self.model_save_path = os.path.join(os.getcwd(),"pretrained_models")

        if self.exp_mode == "sample":
            pass
    