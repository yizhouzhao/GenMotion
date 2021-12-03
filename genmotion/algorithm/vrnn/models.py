import torch
import torch.nn as nn
import torch.utils
import torch.utils.data

class VRNN(nn.Module):
    def __init__(self, opt):
        # x_dim, h_dim, z_dim, n_layers, device, bias=False, resample_output = False
        super(VRNN, self).__init__()
        self.x_dim = opt["input_dim"]
        self.h_dim = opt["hidden_dim"]
        self.z_dim = opt["z_dim"]
        self.n_layers = opt["n_layers"]
        self.resample_output = False
        self.bias = False

        # feature-extracting transformations
        self.phi_x = nn.Sequential(
            nn.Linear(self.x_dim, self.h_dim),
            nn.ReLU(),
            nn.Linear(self.h_dim, self.h_dim),
            nn.ReLU())
        self.phi_z = nn.Sequential(
            nn.Linear(self.z_dim, self.h_dim),
            nn.ReLU())

        # encoder
        self.enc = nn.Sequential(
            nn.Linear(self.h_dim + self.h_dim, self.h_dim),
            nn.ReLU(),
            nn.Linear(self.h_dim, self.h_dim),
            nn.ReLU())
        self.enc_mean = nn.Linear(self.h_dim, self.z_dim)
        self.enc_std = nn.Sequential(
            nn.Linear(self.h_dim, self.z_dim),
            nn.Softplus())

        # prior
        self.prior = nn.Sequential(
            nn.Linear(self.h_dim, self.h_dim),
            nn.ReLU())
        self.prior_mean = nn.Linear(self.h_dim, self.z_dim)
        self.prior_std = nn.Sequential(
            nn.Linear(self.h_dim, self.z_dim),
            nn.Softplus())

        # decoder
        self.dec = nn.Sequential(
            nn.Linear(self.h_dim + self.h_dim, self.h_dim),
            nn.ReLU(),
            nn.Linear(self.h_dim, self.h_dim),
            nn.ReLU())
        self.dec_std = nn.Sequential(
            nn.Linear(self.h_dim, self.x_dim),
            nn.Softplus())
        self.dec_mean = nn.Sequential(
            nn.Linear(self.h_dim, self.x_dim))

        # recurrence
        self.rnn = nn.GRU(self.h_dim + self.h_dim, self.h_dim, self.n_layers, self.bias)

    def forward(self, x, x_padding):
        all_enc_mean, all_enc_std = [], []
        all_dec_mean, all_dec_std = [], []
        kld_loss = 0
        mse_loss = 0

        h = nn.Parameter(torch.zeros(self.n_layers, x.size(1), self.h_dim), requires_grad=True)
        h = h.to(self.device)
        for t in range(x.size(0)):
            phi_x_t = self.phi_x(x[t])
            padding = x_padding[t]

            # encoder
            enc_t = self.enc(torch.cat([phi_x_t, h[-1]], 1))
            enc_mean_t = self.enc_mean(enc_t)
            enc_std_t = self.enc_std(enc_t)

            # prior
            prior_t = self.prior(h[-1])
            prior_mean_t = self.prior_mean(prior_t)
            prior_std_t = self.prior_std(prior_t)

            # sampling and reparameterization
            z_t = self._reparameterized_sample(enc_mean_t, enc_std_t)
            phi_z_t = self.phi_z(z_t)

            # decoder
            dec_t = self.dec(torch.cat([phi_z_t, h[-1]], 1))
            dec_mean_t = self.dec_mean(dec_t)
            dec_std_t = self.dec_std(dec_t)

            # output sample
            #if self.resample_output:
            #    out_t = self._reparameterized_sample(dec_mean_t, dec_std_t)
            #else:
            out_t = dec_mean_t

            # recurrence
            _, h = self.rnn(torch.cat([phi_x_t, phi_z_t], 1).unsqueeze(0), h)

            # computing losses
            kld_loss += self._kld_gauss(enc_mean_t, enc_std_t, prior_mean_t, prior_std_t, padding)
            mse_loss += self._mse_loss(out_t, x[t], padding)

            all_enc_std.append(enc_std_t)
            all_enc_mean.append(enc_mean_t)
            all_dec_mean.append(dec_mean_t)
            all_dec_std.append(dec_std_t)

            # if random.random() < 0.002:
            #     print("time:", t)
            #     print("random output enc: ", enc_mean_t.data, enc_std_t.data, padding.data)
            #     print("random output prior: ", prior_mean_t.data, prior_std_t.data)
            #     print("random output data: ", out_t.data, x[t].data, padding.data)

        return kld_loss, mse_loss, \
               (all_enc_mean, all_enc_std), \
               (all_dec_mean, all_dec_std)

    def reconstruct(self, x):
        '''
        reconstruct one sample
        :param x: [seq_len, 1, dim]
        :return:x': [seq, 1, dim]
        '''
        h = nn.Parameter(torch.zeros(self.n_layers, x.size(1), self.h_dim), requires_grad=True)
        h = h.to(self.device)

        x_rec = torch.zeros_like(x)
        for t in range(x.size(0)):
            phi_x_t = self.phi_x(x[t])

            # encoder
            enc_t = self.enc(torch.cat([phi_x_t, h[-1]], 1))
            enc_mean_t = self.enc_mean(enc_t)
            enc_std_t = self.enc_std(enc_t)

            # sampling and reparameterization
            z_t = self._reparameterized_sample(enc_mean_t, enc_std_t)
            phi_z_t = self.phi_z(z_t)

            # decoder
            dec_t = self.dec(torch.cat([phi_z_t, h[-1]], 1))
            dec_mean_t = self.dec_mean(dec_t)
            #dec_std_t = self.dec_std(dec_t)

            x_rec[t] = dec_mean_t

            # recurrence
            _, h = self.rnn(torch.cat([phi_x_t, phi_z_t], 1).unsqueeze(0), h)

        return x_rec

    def sample(self, seq_len):
        sample = torch.zeros(seq_len, self.x_dim)
        sample = sample.to(self.device)

        h = nn.Parameter(torch.randn(self.n_layers, 1, self.h_dim), requires_grad=True)
        h = h.to(self.device)
        for t in range(seq_len):
            # prior
            prior_t = self.prior(h[-1])
            prior_mean_t = self.prior_mean(prior_t)
            prior_std_t = self.prior_std(prior_t)

            # sampling and reparameterization
            z_t = self._reparameterized_sample(prior_mean_t, prior_std_t)
            phi_z_t = self.phi_z(z_t)

            # decoder
            dec_t = self.dec(torch.cat([phi_z_t, h[-1]], 1))
            dec_mean_t = self.dec_mean(dec_t)
            dec_std_t = self.dec_std(dec_t)

            phi_x_t = self.phi_x(dec_mean_t)

            # recurrence
            _, h = self.rnn(torch.cat([phi_x_t, phi_z_t], 1).unsqueeze(0), h)

            sample[t] = dec_mean_t.data

        return sample

    def sample_next(self, x):
        h = nn.Parameter(torch.zeros(self.n_layers, 1, self.h_dim), requires_grad=True)
        h = h.to(self.device)

        seq_len = x.size(0)
        for t in range(seq_len):
            phi_x_t = self.phi_x(x[t])

            # encoder
            enc_t = self.enc(torch.cat([phi_x_t, h[-1]], 1))
            enc_mean_t = self.enc_mean(enc_t)
            enc_std_t = self.enc_std(enc_t)

            # sampling and reparameterization
            z_t = self._reparameterized_sample(enc_mean_t, enc_std_t)
            phi_z_t = self.phi_z(z_t)

            # recurrence
            _, h = self.rnn(torch.cat([phi_x_t, phi_z_t], 1).unsqueeze(0), h)

        # prior
        prior_t = self.prior(h[-1])
        prior_mean_t = self.prior_mean(prior_t)
        prior_std_t = self.prior_std(prior_t)

        # sampling and reparameterization
        z_t = self._reparameterized_sample(prior_mean_t, prior_std_t)
        phi_z_t = self.phi_z(z_t)

        # decoder
        dec_t = self.dec(torch.cat([phi_z_t, h[-1]], 1))
        dec_mean_t = self.dec_mean(dec_t)
        dec_std_t = self.dec_std(dec_t)

        # output sample
        # next_sample = self._reparameterized_sample(dec_mean_t, dec_std_t)
        next_sample = dec_mean_t

        return next_sample

    def _reparameterized_sample(self, mean, std):
        """using std to sample"""
        eps = torch.FloatTensor(std.size()).normal_()
        eps = nn.Parameter(eps, requires_grad=False).to(self.device)
        return eps * std + mean

    def _kld_gauss(self, mean_1, std_1, mean_2, std_2, padding):
        """Using std to compute KLD"""

        kld_element = (2 * torch.log(std_2) - 2 * torch.log(std_1) +
                       (std_1.pow(2) + (mean_1 - mean_2).pow(2)) /
                       std_2.pow(2) - 1)
        return torch.mean(0.5 * torch.sum(kld_element, dim=1) * padding)

    def _mse_loss(self, pred_x, ori_x, padding):
        """calculate mean square error for reconstruction"""
        #print("VRNN", pred_x.shape)
        return torch.mean(torch.sum((pred_x - ori_x)**2, dim=1) * padding)