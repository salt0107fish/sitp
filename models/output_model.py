import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


class OutputModel(nn.Module):
    def __init__(self, d_k=64, predict_yaw=False):
        super(OutputModel, self).__init__()
        self.d_k = d_k
        self.predict_yaw = predict_yaw
        out_len = 5
        if predict_yaw:
            out_len = 6

        init_ = lambda m: init(m, nn.init.xavier_normal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2))


        self.observation_model = nn.Sequential(
            init_(nn.Linear(self.d_k, self.d_k)), nn.ReLU(),
            init_(nn.Linear(self.d_k, self.d_k)), nn.ReLU(),
            init_(nn.Linear(self.d_k, out_len))
        )

        self.min_stdev = 0.01
        self.corr_t = 0.9


    def forward(self, agent_latent_state):
        T = agent_latent_state.shape[0] 
        BK = agent_latent_state.shape[1] 

        pred_obs = self.observation_model(agent_latent_state.reshape(-1, self.d_k)).reshape(T, BK, -1) 

        x_mean = pred_obs[:, :, 0]
        y_mean = pred_obs[:, :, 1]
        

        x_sigma = F.softplus(pred_obs[:, :, 2]) + self.min_stdev
        y_sigma = F.softplus(pred_obs[:, :, 3]) + self.min_stdev
        rho = torch.tanh(pred_obs[:, :, 4]) * self.corr_t  

        yaws = pred_obs[:, :, 5]  
        return torch.stack([x_mean, y_mean, x_sigma, y_sigma, rho, yaws], dim=2)


