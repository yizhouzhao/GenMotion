import torch
import torch.nn as nn

hidden_dim = 256

class Encoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim
        self.linear = nn.Linear(input_dim, hidden_dim)

    def forward(self, x):
        re


class EncoderRecurrentDecoder(nn.Module):
    def __init__(self):
        super().__init__()