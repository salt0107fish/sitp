import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.context_encoders import MapEncoderPtsMA, PositionalEncoding
from models.output_model import OutputModel


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module



class SITP(nn.Module):
    def __init__(self, args, d_k=128, _M=5, c=5, T=30, L_enc=1, dropout=0.0,
                 k_attr=2, o_attr=3, map_attr=3, num_heads=16, L_dec=1,
                 tx_hidden_size=384, 
                 use_map_lanes=False, num_agent_types=None, predict_yaw=False):
        super(SITP, self).__init__()

        init_ = lambda m: init(m, nn.init.xavier_normal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2))

        # ============================== Basic config ==============================
        self.args = args
        self.k_attr = k_attr 
        self.o_attr = o_attr 
        self.map_attr = map_attr
        self.d_k = d_k
        self._M = _M 
        self.c = c
        self.s = 3 # for strategy
        self.T = T
        self.L_enc = L_enc
        self.dropout = dropout
        self.num_heads = num_heads
        self.L_dec = L_dec
        self.tx_hidden_size = tx_hidden_size
        self.use_map_lanes = use_map_lanes
        self.predict_yaw = predict_yaw

        
        # ============================== History Encoding ==============================
        ## INPUT ENCODERS
        self.agents_dynamic_encoder = nn.Sequential(init_(nn.Linear(self.k_attr, self.d_k)))

        self.social_attn_layers = []
        self.temporal_attn_layers = []

        for _ in range(self.L_enc):
            tx_encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_k, nhead=self.num_heads,
                                                          dropout=self.dropout, dim_feedforward=self.tx_hidden_size)
            self.temporal_attn_layers.append(nn.TransformerEncoder(tx_encoder_layer, num_layers=2))

            tx_encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_k, nhead=self.num_heads,
                                                          dropout=self.dropout, dim_feedforward=self.tx_hidden_size)
            self.social_attn_layers.append(nn.TransformerEncoder(tx_encoder_layer, num_layers=1))

        self.temporal_attn_layers = nn.ModuleList(self.temporal_attn_layers)
        self.social_attn_layers = nn.ModuleList(self.social_attn_layers)
        self.pos_encoder = PositionalEncoding(self.d_k, dropout=0.0)

        if self.use_map_lanes:
            self.map_encoder = MapEncoderPtsMA(d_k=self.d_k, map_attr=self.map_attr, dropout=self.dropout)
            self.map_attn_layers = nn.MultiheadAttention(self.d_k, num_heads=self.num_heads, dropout=self.dropout)
            self.map_attn_layers_init = nn.MultiheadAttention(self.d_k, num_heads=self.num_heads, dropout=self.dropout)

        # ============================== Interaction Decoding ==============================

        self.emb_agent_types = nn.Sequential(init_(nn.Linear(num_agent_types, self.d_k)))
        self.dec_agenttypes_encoder = nn.Sequential(
            init_(nn.Linear(2 * self.d_k, self.d_k)), nn.ReLU(),
            init_(nn.Linear(self.d_k, self.d_k))
        )

        self.Qt2 = nn.Parameter(torch.Tensor(self.T, 1, self.c, 1, self.d_k), requires_grad=True) # change:self.Q
        nn.init.xavier_uniform_(self.Qt2)

        self.social_attn_decoder_layers = []
        self.temporal_attn_decoder_layers = []
        self.mode_attn_decoder_layers = []
        self.social_attn_decoder_layers_last = []
        self.temporal_attn_decoder_layers_last = []

        for _ in range(self.L_dec):
            tx_decoder_layer = nn.TransformerDecoderLayer(d_model=self.d_k, nhead=self.num_heads,
                                                          dropout=self.dropout, dim_feedforward=self.tx_hidden_size)
            self.temporal_attn_decoder_layers.append(nn.TransformerDecoder(tx_decoder_layer, num_layers=2))
            tx_encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_k, nhead=self.num_heads,
                                                          dropout=self.dropout, dim_feedforward=self.tx_hidden_size)
            self.social_attn_decoder_layers.append(nn.TransformerEncoder(tx_encoder_layer, num_layers=1))

            mode_encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_k, nhead=self.num_heads,
                                                            dropout=self.dropout, dim_feedforward=self.tx_hidden_size)
            self.mode_attn_decoder_layers.append(nn.TransformerEncoder(mode_encoder_layer, num_layers=1))

            self.temporal_attn_decoder_layers_last.append(nn.TransformerDecoder(tx_decoder_layer, num_layers=2))
            self.social_attn_decoder_layers_last.append(nn.TransformerEncoder(tx_encoder_layer, num_layers=1))

        self.temporal_attn_decoder_layers = nn.ModuleList(self.temporal_attn_decoder_layers)
        self.social_attn_decoder_layers = nn.ModuleList(self.social_attn_decoder_layers)
        self.mode_attn_decoder_layers = nn.ModuleList(self.mode_attn_decoder_layers)
        self.temporal_attn_decoder_layers_last = nn.ModuleList(self.temporal_attn_decoder_layers_last)
        self.social_attn_decoder_layers_last = nn.ModuleList(self.social_attn_decoder_layers_last)
        
    
        # ============================== Ouput model ==============================
        self.output_model = OutputModel(d_k=self.d_k, predict_yaw=self.predict_yaw)
        self.output_model_init = OutputModel(d_k=self.d_k, predict_yaw=self.predict_yaw)

        # ============================== Strategy & mode ==============================
        self.Qt4 = nn.Parameter(torch.Tensor(c, 1, 1, d_k), requires_grad=True)  # change:self.P
        nn.init.xavier_uniform_(self.Qt4)
        
        if self.use_map_lanes:
            self.mode_map_attn = nn.MultiheadAttention(self.d_k, num_heads=self.num_heads, dropout=self.dropout)
            
        self.prob_decoder = nn.MultiheadAttention(self.d_k, num_heads=self.num_heads, dropout=self.dropout)
        self.prob_predictor = init_(nn.Linear(self.d_k, 1))
        
        if args.strategy:
            self.Qt3 = nn.Parameter(torch.Tensor(self.s, 1, 1, d_k), requires_grad=True)  # change:self.S
            nn.init.xavier_uniform_(self.Qt3)
            self.strategy_decoder = nn.MultiheadAttention(self.d_k, num_heads=self.num_heads, dropout=self.dropout)
            self.strategy_predictor = init_(nn.Linear(self.d_k, 1))
            self.strategy_map_attn = nn.MultiheadAttention(self.d_k, num_heads=self.num_heads, dropout=self.dropout)

        self.train()

    def generate_decoder_mask(self, seq_len, device):
        subsequent_mask = (torch.triu(torch.ones((seq_len, seq_len), device=device), diagonal=1)).bool()
        return subsequent_mask

    def process_observations(self, ego, agents):
        # ego stuff
        ego_tensor = ego[:, :, :self.k_attr] 
        env_masks = ego[:, :, -1] 

        # Agents stuff
        temp_masks = torch.cat((torch.ones_like(env_masks.unsqueeze(-1)), agents[:, :, :, -1]), dim=-1) # [32,5,9]
        opps_masks = (1.0 - temp_masks).type(torch.BoolTensor).to(agents.device)  # only for agents.
        opps_tensor = agents[:, :, :, :self.k_attr]  # only opponent states

        return ego_tensor, opps_tensor, opps_masks

    def temporal_attn_fn(self, agents_emb, agent_masks, layer):
        T_obs = agents_emb.size(0)
        B = agent_masks.size(0)
        agent_masks = agent_masks.permute(0, 2, 1).reshape(-1, T_obs) # [B,T,N] -> [B,N,T] -> [BN,T]
        agent_masks[:, -1][agent_masks.sum(-1) == T_obs] = False  # Ensure agent's that don't exist don't throw NaNs.
        agents_temp_emb = layer(self.pos_encoder(agents_emb.reshape(T_obs, B * (self._M + 1), -1)),
                                src_key_padding_mask=agent_masks) # [T,B,N,F] -> [T,BN,F] FOR T
        return agents_temp_emb.view(T_obs, B, self._M+1, -1) # [T,BN,F] -> [T,B,N,F]

    def social_attn_fn(self, agents_emb, agent_masks, layer):
        T_obs = agents_emb.size(0)
        B = agent_masks.size(0)
        agents_emb = agents_emb.permute(2, 1, 0, 3).reshape(self._M + 1, B * T_obs, -1) # [T,B,N,F] -> [N,B,T,F] -> [N,BT,F]
        agents_soc_emb = layer(agents_emb, src_key_padding_mask=agent_masks.view(-1, self._M+1))
        agents_soc_emb = agents_soc_emb.view(self._M+1, B, T_obs, -1).permute(2, 1, 0, 3) # [N,B,T,F] -> [T,B,N,F]
        return agents_soc_emb

    def temporal_attn_decoder_fn(self, agents_emb, context, agent_masks, layer):
        T_obs = context.size(0)
        BK = agent_masks.size(0)
        time_masks = self.generate_decoder_mask(seq_len=self.T, device=agents_emb.device)
        agent_masks = agent_masks.permute(0, 2, 1).reshape(-1, T_obs)
        agent_masks[:, -1][agent_masks.sum(-1) == T_obs] = False  # Ensure that agent's that don't exist don't make NaN.
        agents_emb = agents_emb.reshape(self.T, -1, self.d_k)  #(T, BK, N, H) -> [T, BxKxN, H]
        context = context.view(-1, BK*(self._M+1), self.d_k) # (T_in, BK, N, H)-> (T_in, BKN, H)

        agents_temp_emb = layer(agents_emb, context, tgt_mask=time_masks, memory_key_padding_mask=agent_masks) # FOR t
        agents_temp_emb = agents_temp_emb.view(self.T, BK, self._M+1, -1) # [T, BxKxN, H] -> [T,BK,N,H]

        return agents_temp_emb

    def social_attn_decoder_fn(self, agents_emb, agent_masks, layer):
        B = agent_masks.size(0)
        agent_masks = agent_masks[:, -1:].repeat(1, self.T, 1).view(-1, self._M + 1)  # take last timestep of all agents.[BK,1,N]->[BK,T,N]->[BKT,N]
        agents_emb = agents_emb.permute(2, 1, 0, 3).reshape(self._M + 1, B * self.T, -1) # (T, BK, N, H) -> (N, BK, T, H) -> [N.BKT,H]
        agents_soc_emb = layer(agents_emb, src_key_padding_mask=agent_masks)
        agents_soc_emb = agents_soc_emb.view(self._M + 1, B, self.T, -1).permute(2, 1, 0, 3)
        return agents_soc_emb
    
    def mode_attn_decoder_fn(self, agents_emb, agent_masks, layer):
        B = agent_masks.size(0) // self.c # real b
        T,_,N,_ = agents_emb.shape
        # take last timestep of all agents.[BK,1,N]->[BK,T,N]->[B,K,T,N]->[TBN,K]
        agent_masks = agent_masks[:, -1:].repeat(1, self.T, 1).reshape(B,self.c,T,N).permute(2,0,3,1).reshape(-1,self.c)  
        # (T, BK, N, H) ->  [K,TBN,H]
        agents_emb = agents_emb.reshape(T,B,self.c,N,-1).permute(2,0,1,3,4).reshape(self.c,T*B*N,-1) 
        agents_soc_emb = layer(agents_emb, 
                               # src_key_padding_mask=agent_masks
                               )
        # [K,TBN,H] -> [T,BK,N,H]
        agents_soc_emb = agents_soc_emb.reshape(self.c,T,B,N,-1).permute(1,2,0,3,4).reshape(T,B*self.c,N,-1)
        
        return agents_soc_emb

    
    def decoder_query(self,agent_types,B):
        agent_types_features = self.emb_agent_types(agent_types).unsqueeze(1).\
            repeat(1, self.c, 1, 1).view(-1, self._M+1, self.d_k) 
        agent_types_features = agent_types_features.unsqueeze(0).repeat(self.T, 1, 1, 1)

        dec_parameters = self.Qt2.repeat(1, B, 1, self._M+1, 1).view(self.T, B*self.c, self._M+1, -1) 

        dec_parameters = torch.cat((dec_parameters, agent_types_features), dim=-1) 
        dec_parameters = self.dec_agenttypes_encoder(dec_parameters) 

        return dec_parameters
    
    def first_interaction(self,agents_dec_emb,map_features,context,
                                  opps_masks_modes,road_segs_masks,
                                  B):
        for d in range(self.L_dec): 
            if self.use_map_lanes and d == 1:
                agents_dec_emb = agents_dec_emb.reshape(self.T, -1, self.d_k)
                agents_dec_emb_map = self.map_attn_layers(query=agents_dec_emb, key=map_features, value=map_features,
                                                            key_padding_mask=road_segs_masks)[0]
                agents_dec_emb = agents_dec_emb + agents_dec_emb_map
                agents_dec_emb = agents_dec_emb.reshape(self.T, B*self.c, self._M+1, -1)
                agents_dec_emb = self.mode_attn_decoder_fn(agents_dec_emb, 
                                                        opps_masks_modes, 
                                                        layer=self.mode_attn_decoder_layers[d],
                                                        )
            agents_dec_emb = self.temporal_attn_decoder_fn(agents_dec_emb, context, opps_masks_modes, layer=self.temporal_attn_decoder_layers[d])
            agents_dec_emb = self.social_attn_decoder_fn(agents_dec_emb, opps_masks_modes, layer=self.social_attn_decoder_layers[d])

        return agents_dec_emb
    
    def second_interaction(self,agents_dec_emb,map_features,context,
                            opps_masks_modes,road_segs_masks,
                            B):
        if self.use_map_lanes:
            agents_dec_emb = agents_dec_emb.reshape(self.T, -1, self.d_k)
            agents_dec_emb_map = self.map_attn_layers_init(query=agents_dec_emb, key=map_features, value=map_features,
                                                        key_padding_mask=road_segs_masks)[0]
            agents_dec_emb = agents_dec_emb + agents_dec_emb_map
            agents_dec_emb = agents_dec_emb.reshape(self.T, B*self.c, self._M+1, -1)
        agents_dec_emb = self.temporal_attn_decoder_fn(agents_dec_emb, context, opps_masks_modes, layer=self.temporal_attn_decoder_layers_last[1])
        agents_dec_emb = self.social_attn_decoder_fn(agents_dec_emb, opps_masks_modes, layer=self.social_attn_decoder_layers_last[1])
    
        return agents_dec_emb

    def mode_decoding(self,agents_emb,orig_map_features,
                      orig_road_segs_masks,
                      B):
        mode_params_emb = self.Qt4.repeat(1, B, self._M+1, 1).view(self.c, -1, self.d_k) 
        mode_params_emb = self.prob_decoder(query=mode_params_emb, key=agents_emb.reshape(-1, B*(self._M+1), self.d_k),
                                            value=agents_emb.reshape(-1, B*(self._M+1), self.d_k))[0] 
        if self.use_map_lanes:
            orig_map_features = orig_map_features.view(-1, B*(self._M+1), self.d_k)
            orig_road_segs_masks = orig_road_segs_masks.view(B*(self._M+1), -1)
            mode_params_emb = self.mode_map_attn(query=mode_params_emb, key=orig_map_features, value=orig_map_features,
                                                 key_padding_mask=orig_road_segs_masks)[0] + mode_params_emb
            
        mode_probs = self.prob_predictor(mode_params_emb).squeeze(-1).view(self.c, B, self._M+1).sum(2).transpose(0, 1)
        mode_probs = F.softmax(mode_probs, dim=1) 

        return mode_probs

    def strategy_decoding(self,agents_emb,orig_map_features,
                        orig_road_segs_masks,
                        B):
        strategy_params_emb = self.Qt3.repeat(1, B, self._M+1, 1).view(self.s, -1, self.d_k) 
        strategy_params_emb = self.strategy_decoder(query=strategy_params_emb, key=agents_emb.reshape(-1, B*(self._M+1), self.d_k),
                                        value=agents_emb.reshape(-1, B*(self._M+1), self.d_k))[0] 
        if self.use_map_lanes:
            orig_map_features = orig_map_features.view(-1, B*(self._M+1), self.d_k)
            orig_road_segs_masks = orig_road_segs_masks.view(B*(self._M+1), -1)
            strategy_params_emb = self.strategy_map_attn(query=strategy_params_emb, key=orig_map_features, value=orig_map_features,
                                                        key_padding_mask=orig_road_segs_masks)[0] + strategy_params_emb
        

        strategy = self.strategy_predictor(strategy_params_emb).squeeze(-1).view(self.s, B, self._M+1).permute(1,2,0)
        strategy = F.softmax(strategy, dim=-1)# [b,k,3]
    
    def forward(self, ego_in, agents_in, roads, agent_types):
        B = ego_in.size(0) 

        # ============================== Historical Encoding ==============================
        ego_tensor, _agents_tensor, opps_masks = self.process_observations(ego_in, agents_in) 
        agents_tensor = torch.cat((ego_tensor.unsqueeze(2), _agents_tensor), dim=2) 
        agents_emb = self.agents_dynamic_encoder(agents_tensor).permute(1, 0, 2, 3)

        for i in range(self.L_enc): 
            agents_emb = self.temporal_attn_fn(agents_emb, opps_masks, layer=self.temporal_attn_layers[i])
            agents_emb = self.social_attn_fn(agents_emb, opps_masks, layer=self.social_attn_layers[i])

        if self.use_map_lanes: 
            orig_map_features, orig_road_segs_masks = self.map_encoder(roads, agents_emb)
            map_features = orig_map_features.unsqueeze(2).repeat(1, 1, self.c, 1, 1).view(-1, B * self.c * (self._M+1), self.d_k) 
            road_segs_masks = orig_road_segs_masks.unsqueeze(2).repeat(1, self.c, 1, 1).view(B * self.c * (self._M+1), -1) 

        opps_masks_modes = opps_masks.unsqueeze(1).repeat(1, self.c, 1, 1).view(B*self.c, ego_in.shape[1], -1)
        context = agents_emb.unsqueeze(2).repeat(1, 1, self.c, 1, 1)
        context = context.view(ego_in.shape[1], B*self.c, self._M+1, self.d_k)

        # ============================== Interaction Decoding ==============================

        ### decoder query
        agents_dec_emb = self.decoder_query(agent_types,B)
        
        ### First interaction decoding 

        agents_dec_emb = self.first_interaction(agents_dec_emb,map_features,context,
                                            opps_masks_modes,road_segs_masks,
                                            B)

        ### Intermediate Aux. output
        out_dists_init = self.output_model_init(agents_dec_emb.reshape(self.T, -1, self.d_k)) 
        out_dists_init = out_dists_init.reshape(self.T, B, self.c, self._M+1, -1).permute(2, 0, 1, 3, 4) 
        
        ### Second interaction decoding
        agents_dec_emb = self.second_interaction(agents_dec_emb,map_features,context,
                                            opps_masks_modes,road_segs_masks,
                                            B)
        
        out_dists = self.output_model(agents_dec_emb.reshape(self.T, -1, self.d_k))
        out_dists = out_dists.reshape(self.T, B, self.c, self._M+1, -1).permute(2, 0, 1, 3, 4) 

        # ============================== Strategy & mode ==============================
        
        mode_probs = self.mode_decoding(agents_emb,orig_map_features,orig_road_segs_masks,B)
        if self.args.strategy:
            strategy = self.strategy_decoding(agents_emb,orig_map_features,orig_road_segs_masks,B)
        else:
            strategy = None
        
        return out_dists, mode_probs, out_dists_init, strategy

