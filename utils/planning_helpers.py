import torch
from scipy import special
import numpy as np
import torch.distributions as D

import torch.nn.functional as F

def cal_l2_loss(pred, data):
    """
    pred: [6k,15tout,32bs,9n,2]
    data: [32bs,15t,9n,2]
    """
    
    # [1k,6bs]
    K = pred.shape[0]
    pred = pred.permute(0,2,1,3,4) # [k,b,t,n,2]
    data = data.unsqueeze(0) # [1,b,t,n,2]
    fde_loss = torch.norm((pred[:, :, -1, :, :2] - data[:, :, -1]), 2, dim=-1) # [k, b, n]
    ade_loss = (torch.norm((pred - data), 2, dim=-1)).mean(-2) # [k, b, n]

    ade_loss, ade_min_inds = ade_loss.min(dim=0) # [b,n]
    
    total_loss = (fde_loss + ade_loss) # [k, b, n]
    loss, min_inds = total_loss.min(dim=0) # [b,n] min for 6
    
    return ade_min_inds # [b,n]



def cal_safety_cost(pred, gt):

    # gt:[32bs,15tout,9n,2]
    # pred:[6k,15tout,32bs,9n,2]
    
    K,_,_,N,_ = pred.shape # N8
    gt = gt.unsqueeze(0).permute(0,2,1,3,4)
    
    # [6k,15tout,32bs,9n,2]
    distances = torch.linalg.norm(pred.unsqueeze(-2) - gt.unsqueeze(-3), dim=-1) # [k,t,b,n,n],

    diagonal_mask = torch.eye(N, dtype=distances.dtype, device=distances.device).unsqueeze(0).unsqueeze(0).unsqueeze(0)
    distances = torch.where(diagonal_mask > 0, 99999 * torch.ones_like(distances), distances)
    
    distances_inv = 1 / distances # 进行倒数
    distances_inv = distances_inv.mean(1)  # [k,b,n,n],
    
    distance_sum = distances_inv.sum(-1)
    
    return distance_sum # K,bs,n


def cal_effi_cost(pred, gt):
    # ego_candidate: [6k,30t,1bs,6f]
    # ego_truth: [1bs,30,4]

    # gt:[32bs,15tout,9n,2]
    # pred:[6k,15tout,32bs,9n,2]

    eps=0.001

    velocity = (1/(torch.abs(torch.diff(pred, dim=1))+eps)).mean(1).sum(-1)

    return velocity


def cal_effi_cost_old(pred, gt):
    # ego_candidate: [6k,30t,1bs,6f]
    # ego_truth: [1bs,30,4]

    # gt:[32bs,15tout,9n,2]
    # pred:[6k,15tout,32bs,9n,2]

    K,T,_,_,_ = pred.shape
    ego_pos = pred.permute(0,2,1,3,4) # [6k,1bs,30t,9n,2]
    # [32bs,15tout,9n,2] -> [b,1,n,2] -> [1,b,1,n,2] -> [k,b,T,n,2]
    truth_pos = gt[:,-2:-1].unsqueeze(0).repeat(K,1,T,1,1)

    distances = torch.linalg.norm(ego_pos - truth_pos, dim=-1).sum(-2) # [6k,1bs,30t,9n] ->[6k,1bs,9n]

    return distances

def cal_smooth_cost(pred):

    
    ego_v = torch.diff(pred, dim=1)
    ego_a = torch.diff(ego_v, dim=1)
    jerk = torch.abs(torch.diff(ego_a, dim=1)).mean(1).sum(-1) # [6k,t,1bs,n,2] -> [6k,1bs,n]
    
    return jerk

def select_positive_candidate(ego_candidate, ego_truth):
    # ego_candidate: [6k,30t,1bs,62]
    # ego_truth: [1bs,30,2] x+y+yaw?+?

    N = ego_candidate.shape[0] # 6K
    candidate_pos = ego_candidate[...,:2].permute(0,2,1,3) # 6k,1bs,30t,2xy
    truth_pos = ego_truth[...,:2].unsqueeze(0).repeat(N,1,1,1)

    distances = torch.linalg.norm(candidate_pos - truth_pos, dim=-1).mean(-1) # 6,1
    min_value, min_index = torch.min(distances, dim=0) # [1bs]
    
    return min_index

def accelerate_decelerate(tensor, factor):

    velocity = torch.diff(tensor,dim=1) * factor # [K,T-1,BS,N,2]
    cumulative_velocity = torch.zeros_like(tensor).to(tensor.device)
    cumulative_velocity[:,1:] = torch.cumsum(velocity, dim=1)
    position = tensor[:,0:1] + cumulative_velocity
    return position

def expand_sample(pred):
    # pred:[6k,15tout,32bs,9n,2]
    # expand sample
    K,T,B,N,F = pred.shape

    expanded_tensor = torch.empty(0,T,B,N,F).to(pred.device)


    expanded_tensor = torch.cat((expanded_tensor, pred), dim=0)
    accelerated = accelerate_decelerate(pred, factor=1.5)
    expanded_tensor = torch.cat((expanded_tensor, accelerated), dim=0)

    decelerated = accelerate_decelerate(pred, factor=0.5)
    expanded_tensor = torch.cat((expanded_tensor, decelerated), dim=0)

    return expanded_tensor


def cal_pos_by_vel(tensor, velocity):
    # velocity[bs,2] represent the max/min velocity's abs in the scenario
    # tensor [6k,15tout,32bs,9n,2]
    # aim: use the velocity to replace the pred (use +/- from pred)

    K,T,B,N,_ = tensor.shape
    ori_velocity = torch.diff(tensor, dim=1)  # [K,T-1,BS,N,2]
    sign_mask = ori_velocity.sign() # (use +/- from pred)

    velocity = velocity.unsqueeze(1).unsqueeze(0).unsqueeze(0).repeat(K, T-1, 1, N, 1) * sign_mask
    cumulative_velocity = torch.zeros_like(tensor).to(tensor.device)
    cumulative_velocity[:, 1:] = torch.cumsum(velocity, dim=1)
    position = tensor[:, 0:1] + cumulative_velocity
    return position


def expand_sample_case3(pred):
    # pred:[6k,15tout,32bs,9n,2]
    # expand sample
    K,T,B,N,F = pred.shape

    expanded_tensor = torch.empty(0,T,B,N,F).to(pred.device)

    # select max/min velocity
    velocity = torch.abs(torch.diff(pred, dim=1)).permute(0,1,3,2,4).reshape(-1,B,F) # [k,t-1,bs,n,2]
    velocity_min = torch.min(velocity,dim=0)[0] # [bs,2]
    velocity_max = torch.max(velocity, dim=0)[0]  # [bs,2]
    expanded_tensor = torch.cat((expanded_tensor, pred), dim=0)

    accelerated = cal_pos_by_vel(pred, velocity=velocity_max)
    expanded_tensor = torch.cat((expanded_tensor, accelerated), dim=0)

    decelerated = cal_pos_by_vel(pred, velocity=velocity_min)
    expanded_tensor = torch.cat((expanded_tensor, decelerated), dim=0)

    return expanded_tensor

def cal_pl_loss(pred,gt,pred_probs,strategy):

    # gt:[32bs,15tout,9n,2]
    # pred:[6k,15tout,32bs,9n,2]
    # pred_probs: [bs, 6k]
    # strategy [b,n,3]
    
    # ego_truth = gt[:,:,0] # [32bs,15tout,2]
    # agents_pred = pred[:,:,:,1:].permute(2,1,3,0,4)
    # ego_candidate = pred[:,:,:,0] # [6k,15t,32bs,2f]


    pred = expand_sample(pred)

    Cn = strategy.shape[-1]
    K, _, BS, N, _ = pred.shape


    best_traj_idx = cal_l2_loss(pred, gt) # [1bs, n] 


    all_cost = torch.zeros((Cn, K, BS, N)).to(strategy.device) # cn,6k,1bs
    all_cost[0] = cal_safety_cost(pred, gt) # 6k,b,n
    all_cost[1] = cal_effi_cost(pred, gt)
    all_cost[2] = cal_smooth_cost(pred)
    all_cost = torch.nn.functional.normalize(all_cost, dim=1)
    # strategy [b,n,3] allcost[3,k,bs,n]
    # [k,b,n,3] * [3,k,b,n] -> [k,b,n,3] -> [k,b,n] -> [b,n,k]
    cost = torch.mul(strategy.unsqueeze(0).repeat(K,1,1,1), all_cost.permute(1,2,3,0)).sum(-1).permute(1,2,0)

    label = torch.ones_like(cost)
    label[torch.arange(BS)[:, None], torch.arange(N), best_traj_idx] = 0
    strategy_loss = F.cross_entropy(cost, label)
    select_id = cost[:, :, :K // 3].argmin(dim=-1)  
    
    return strategy_loss, best_traj_idx, select_id, all_cost