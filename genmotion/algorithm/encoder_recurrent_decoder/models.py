import torch
import torch.nn as nn

encoder_hidden_dim = 500
lstm_hidden_dim = 1000
decoder_hidden_dim = 100
lstm_layer_num = 2

class Encoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim
        self.linear = nn.Sequential(
            nn.Linear(input_dim, encoder_hidden_dim),
            nn.ReLU(),
            nn.Linear(encoder_hidden_dim, lstm_hidden_dim),
        )
        
    def forward(self, x):
        return self.linear(x)


class Decoder(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        self.output_dim = output_dim
        self.linear = nn.Sequential(
            nn.Linear(lstm_hidden_dim, encoder_hidden_dim),
            nn.ReLU(),
            nn.Linear(encoder_hidden_dim, decoder_hidden_dim),
            nn.ReLU(),
            nn.Linear(decoder_hidden_dim, output_dim),
        )
        
    def forward(self, x):
        return self.linear(x)


class EncoderRecurrentDecoder(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.encoder = Encoder(opt.get("input_dim"))
        self.decoder = Decoder(opt.get("input_dim"))
        self.rnn = nn.LSTM(lstm_hidden_dim, lstm_hidden_dim, lstm_layer_num, batch_first = True)

    def forward(self, x):
        x = self.encoder(x)
        
        #h0 = torch.zeros(lstm_layer_num, x.shape(0), lstm_hidden_dim).to(x.device)
        #c0 = torch.zeros(lstm_layer_num, x.shape(0), lstm_hidden_dim).to(x.device)
        
        x, _ = self.rnn(x)
        x = self.decoder(x)

        return x