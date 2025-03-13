import numpy as np
import torch
import torch.nn as nn
import math


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=20):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class MapEncoderPtsMA(nn.Module):
    def __init__(self, d_k, map_attr=3, dropout=0.1):
        super(MapEncoderPtsMA, self).__init__()
        self.dropout = dropout
        self.d_k = d_k
        init_ = lambda m: init(m, nn.init.xavier_normal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2))

        self.map_attr = map_attr

        # Seed parameters for the map
        self.map_seeds = nn.Parameter(torch.Tensor(1, 1, self.d_k), requires_grad=True)
        nn.init.xavier_uniform_(self.map_seeds)

        self.road_pts_lin = nn.Sequential(init_(nn.Linear(self.map_attr, self.d_k)))
        self.road_pts_attn_layer = nn.MultiheadAttention(self.d_k, num_heads=8, dropout=self.dropout)
        self.norm1 = nn.LayerNorm(self.d_k, eps=1e-5)
        self.norm2 = nn.LayerNorm(self.d_k, eps=1e-5)
        self.map_feats = nn.Sequential(
            init_(nn.Linear(self.d_k, self.d_k*3)), nn.ReLU(), nn.Dropout(self.dropout),
            init_(nn.Linear(self.d_k*3, self.d_k)),
        )

    def get_road_pts_mask(self, roads):
        road_segment_mask = torch.sum(roads[:, :, :, :, -1], dim=3) == 0
        road_pts_mask = (1.0 - roads[:, :, :, :, -1]).type(torch.BoolTensor).to(roads.device).view(-1, roads.shape[3])

        # The next lines ensure that we do not obtain NaNs during training for missing agents or for empty roads.
        road_pts_mask[:, 0][road_pts_mask.sum(-1) == roads.shape[3]] = False  # for empty agents
        road_segment_mask[:, :, 0][road_segment_mask.sum(-1) == road_segment_mask.shape[2]] = False  # for empty roads
        return road_segment_mask, road_pts_mask

    def forward(self, roads, agents_emb):
        B = roads.shape[0]
        M = roads.shape[1]
        S = roads.shape[2]
        P = roads.shape[3]
        road_segment_mask, road_pts_mask = self.get_road_pts_mask(roads)
        road_pts_feats = self.road_pts_lin(roads[:, :, :, :, :self.map_attr]).view(B*M*S, P, -1).permute(1, 0, 2)

        map_seeds = self.map_seeds.repeat(1, B * M * S, 1)
        road_seg_emb = self.road_pts_attn_layer(query=map_seeds, key=road_pts_feats, value=road_pts_feats,
                                                key_padding_mask=road_pts_mask)[0]
        road_seg_emb = self.norm1(road_seg_emb)
        road_seg_emb2 = road_seg_emb + self.map_feats(road_seg_emb)
        road_seg_emb2 = self.norm2(road_seg_emb2)
        road_seg_emb = road_seg_emb2.view(B, M, S, -1)

        return road_seg_emb.permute(2, 0, 1, 3), road_segment_mask
