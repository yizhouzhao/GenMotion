import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from IPython.display import clear_output
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, dataloader
from torch.nn.modules.conv import ConvTranspose1d
from torch.nn import Module, Conv1d, Sequential, Dropout, MaxPool1d


class MotionLSTM(Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.W = nn.Parameter(torch.Tensor(input_size, hidden_size * 4))
        self.U = nn.Parameter(torch.Tensor(hidden_size, hidden_size * 4))
        self.bias = nn.Parameter(torch.Tensor(hidden_size * 4))
        self.init_weights()
                
    def init_weights(self):
        stdv = 1.0 / (self.hidden_size ** 0.5)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
         
    def forward(self, x, init_states=None, isSampling=False, output_frame = 0):
        """Assumes x is of shape (batch, feature, sequence)"""
        batch_size, _, sequence_len = x.size()
        hidden_seq = []
        if init_states is None:
            h_t, c_t = (torch.zeros(batch_size, self.hidden_size).to(x.device), 
                        torch.zeros(batch_size, self.hidden_size).to(x.device))
        else:
            h_t, c_t = init_states
         
        HS = self.hidden_size
        for t in range(sequence_len + output_frame):
            x_t = x[:, :, t]
            # batch the computations into a single matrix multiplication
            gates = x_t @ self.W + h_t @ self.U + self.bias
            i_t, f_t, g_t, o_t = (
                torch.sigmoid(gates[:, :HS]), # input
                torch.sigmoid(gates[:, HS:HS*2]), # forget
                torch.tanh(gates[:, HS*2:HS*3]),
                torch.sigmoid(gates[:, HS*3:]), # output
            )
            c_t = f_t * c_t + i_t * g_t if isSampling else f_t * x_t + i_t * g_t
            h_t = o_t * torch.tanh(c_t)
            hidden_seq.append(h_t.unsqueeze(2))
        hidden_seq = torch.cat(hidden_seq, dim=2)
        hidden_seq = hidden_seq.contiguous()
        return hidden_seq, (h_t, c_t)

    def sampling(self, x, output_frame, init_states=None):
        return self.forward(x, init_states, True, output_frame)[0]

class VAE_LSTM(Module):   
    def __init__(self, joint_size: int, input_frame: int):
        super(VAE_LSTM, self).__init__()

        self.joint_size = joint_size 
        self.input_frame = input_frame
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.encoder1 = Sequential(
            Dropout(0.2),
            Conv1d(in_channels=joint_size * 3, out_channels=64, kernel_size=25, padding=12),
            MaxPool1d(kernel_size=2)
        )

        self.encoder2 = Sequential(
            Dropout(0.2),
            Conv1d(in_channels=64, out_channels=256, kernel_size=25, padding=12),
            MaxPool1d(kernel_size=2)
        )
        
        self.rnn = MotionLSTM(input_size=128, hidden_size=128)

        self.decoder1 = Sequential(
            ConvTranspose1d(in_channels=128,  out_channels=128, kernel_size=2, stride=2),   
            Dropout(0.2),
            Conv1d(in_channels=128, out_channels=128, kernel_size=25, padding=12),
        )

        self.decoder2 = Sequential(
            ConvTranspose1d(in_channels=128, out_channels=128, kernel_size=2, stride=2),
            Dropout(0.2),
            Conv1d(in_channels=128, out_channels=joint_size * 3, kernel_size=25, padding=12),
        )
  
    def forward(self, x):
        x = F.elu(self.encoder1(x))
        x = F.elu(self.encoder2(x))

        mu, log_var = x[:, 0::2, :], x[:, 1::2, :]
        z = torch.randn(size=mu.shape)
        h = mu + torch.exp(0.5 * log_var) * z
        m, _ = self.rnn(h)

        x_hat = F.elu(self.decoder1(m))
        x_hat = F.elu(self.decoder2(x_hat))
        return x_hat, mu, log_var

    def generate(self, x, output_frame):
        x = F.elu(self.encoder1(x))
        x = F.elu(self.encoder1(x))

        mu, log_var = x[:, 0::2, :], x[:, 1::2, :]
        z = torch.randn(size=mu.shape)
        h = mu + torch.exp(0.5 * log_var) * z
        m = self.rnn.sampling(h, output_frame)

        x_hat = F.elu(self.decoder1(m))
        x_hat = F.elu(self.decoder2(x_hat))
        return x_hat


    def train(self, lr: float, num_epochs: int, train_loader: dataloader, display="visual"):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        loss_lst = []

        for _ in tqdm(range(num_epochs)):
            for data_dict in train_loader:
                data = torch.concat([d for d in data_dict.values()], dim=1)
                data = data.to(self.device).to(torch.float32)

                out, mu, log_var = self.forward(data)

                kl_divergence = 0.5 * torch.sum(-1 - log_var + mu.pow(2) + log_var.exp())
                loss = F.gaussian_nll_loss(out, data, torch.ones(out.shape, requires_grad=True)) + kl_divergence

                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            with torch.no_grad():
                loss_lst.append(loss)
                clear_output(wait=True)
                plt.xlabel("# of Epoch")
                plt.ylabel("Loss") 
                plt.legend()
                plt.plot(loss_lst)
                plt.show()
        print(f"Final loss: {loss}")


class MotionDataset(Dataset):
    """Motion"""

    def __init__(self, root_dir: str, fetch: Optional[Callable] = None, transform: Optional[Callable] = None):
        """
        Args:
            root_dir (string): Directory with all the motion npz files.
            fetch (callable, optional): Optional function to specify how data is fetched
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform

        # get all file name
        if fetch:
            self.file_name = fetch(root_dir)
        else:
            self.file_name = []
            for r, d, f in os.walk(self.root_dir):
                for file in f:
                    self.file_name.append(r + "/" + file)

        self.sanity_check()                

    def __len__(self):
        return len(self.file_name)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

            sample = []
            for i in idx: 
                if self.transform:
                    sample.append(self.transform(np.load(self.file_name[i]), allow_pickle=True))
            
        else:
            sample = np.load(self.file_name[idx], allow_pickle=True)
            if self.transform:
                sample = self.transform(sample)

        return sample

    def sanity_check(self):
        for file_path in self.file_name:
            data = np.load(file_path)
            for field in ["trans", "root_orient", "poses"]:
                if field not in data:
                    self.file_name.remove(file_path)
                    break