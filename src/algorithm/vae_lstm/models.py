from typing import Tuple
from tqdm import tqdm
import matplotlib.pyplot as plt
from IPython.display import clear_output
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import dataloader
from torch.nn.modules.conv import ConvTranspose1d
from torch.nn import Module, Conv1d, Sequential, Dropout, MaxPool1d


class MotionLSTM(Module):
    """A Motion LSTM model for recurrent generation of motion sequence
    
    :param input_size: size of input vector
    :type input_size: int

    :param hidden_size: size of hidden state vector
    :type hidden_size: int
    """

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.W = nn.Parameter(torch.Tensor(input_size, hidden_size * 4))
        self.U = nn.Parameter(torch.Tensor(hidden_size, hidden_size * 4))
        self.bias = nn.Parameter(torch.Tensor(hidden_size * 4))
        
        stdv = 1.0 / (self.hidden_size ** 0.5)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
        
         
    def forward(self, x, init_states: Tuple[torch.tensor, torch.tensor] = None, isSampling: bool = False, num_adding_frame: int = 0):
        """Forward method for the model
        
        :param init_states: initial states for hidden and cell state vector
        :type init_states: Tuple[torch.tensor, torch.tensor]

        :param isSampling: whether we are doing sampling or training
        :type isSampling: bool

        :param num_adding_frame: number of frames to be additonally sampled (only useful in sampling mode, for training mode it should be set to 0)
        :type num_adding_frame: int
        """
        batch_size, _, sequence_len = x.size()
        hidden_seq = []
        if init_states is None:
            h_t, c_t = (torch.zeros(batch_size, self.hidden_size).to(x.device), 
                        torch.zeros(batch_size, self.hidden_size).to(x.device))
        else:
            h_t, c_t = init_states
         
        HS = self.hidden_size
        for t in range(sequence_len + num_adding_frame):
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

    def sampling(self, x: torch.tensor, num_adding_frame: int, init_states: Tuple[torch.tensor, torch.tensor] = None):
        """Sampling output given input sequence
        
        :param x: input sequence
        :type x: torch.tensor

        :param num_adding_frame: whether we are doing sampling or training
        :type num_adding_frame: int

        :param init_states: initial states for hidden and cell state vector
        :type init_states: Tuple[torch.tensor, torch.tensor]
        """
        return self.forward(x, init_states, True, num_adding_frame)[0]

class VAE_LSTM(Module): 
    """VAE-LSTM model for motion sequence generation
    
    :param num_joints: number of joints in model 
    :type num_joints: int

    :param input_frame: number of frames to be additonally sampled
    :type input_frame: int    
    """

    def __init__(self, num_joints: int, input_frame: int):
        super(VAE_LSTM, self).__init__()

        self.num_joints = num_joints 
        self.input_frame = input_frame
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.encoder1 = Sequential(
            Dropout(0.2),
            Conv1d(in_channels=num_joints * 3, out_channels=64, kernel_size=25, padding=12),
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
            Conv1d(in_channels=128, out_channels=num_joints * 3, kernel_size=25, padding=12),
        )
  
    def forward(self, x):
        """Forward method for the model
        
        :param x: input motion sequence
        :type x: torch.tensor
        """
        x = F.elu(self.encoder1(x))
        x = F.elu(self.encoder2(x))

        mu, log_var = x[:, 0::2, :], x[:, 1::2, :]
        z = torch.randn(size=mu.shape)
        h = mu + torch.exp(0.5 * log_var) * z
        m, _ = self.rnn(h)

        x_hat = F.elu(self.decoder1(m))
        x_hat = F.elu(self.decoder2(x_hat))
        return x_hat, mu, log_var

    def sampling(self, x, num_adding_frame):
        """Sampling output given input sequence
        
        :param num_adding_frame: number of frames to be additionally sampled 
        :type num_adding_frame: int

        :param num_adding_frame: whether we are doing sampling or training
        :type num_adding_frame: int

        :param init_states: initial states for hidden and cell state vector
        :type init_states: Tuple[torch.tensor, torch.tensor]]

        Note that due to pooling/unpooling, the actually frame additionally generated will be 4 * num_adding_frame
        """
        x = F.elu(self.encoder1(x))
        x = F.elu(self.encoder1(x))

        mu, log_var = x[:, 0::2, :], x[:, 1::2, :]
        z = torch.randn(size=mu.shape)
        h = mu + torch.exp(0.5 * log_var) * z
        m = self.rnn.sampling(h, num_adding_frame)

        x_hat = F.elu(self.decoder1(m))
        x_hat = F.elu(self.decoder2(x_hat))
        return x_hat


    def train(self, lr: float, num_epochs: int, train_loader: dataloader):
        """train the data and output the loss
        
        :param lr: learning rate
        :type lr: float

        :param num_epochs: number of training epochs
        :type num_epochs: int

        :param train_loader: PyTorch data loader
        :type train_loader: torch.utils.data.dataloader
        """
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
