import torch
from scipy import special
import numpy as np
import torch.distributions as D
from torch.distributions import MultivariateNormal, Laplace

import torch.nn.functional as F

from utils.planning_helpers import cal_pl_loss

def make_2d_rotation_matrix(angle_in_radians: torch.Tensor) -> torch.Tensor:
    """
    Makes rotation matrix to rotate point in x-y plane counterclockwise
    by angle_in_radians. [32,9]
    """
    bs,n = angle_in_radians.shape
    cos_theta = torch.cos(angle_in_radians)
    sin_theta = torch.sin(angle_in_radians)
    return torch.stack([
        cos_theta, -sin_theta,
        sin_theta, cos_theta
    ], dim=-1).reshape(bs,n,2,2)

# global
def convert_pred_global(local_coords, translations):
    # pred: [6,15tout,32bs,9n,2xy]
    # translations: [32bs,9n,3xyyaw]
    """
    """
    num_modes, time_out, batch_size, num_points, _ = local_coords.shape
    # Create the rotation matrices for each point
    angles = -translations[..., 2]
    rotations = make_2d_rotation_matrix(angles).unsqueeze(1)  # [batch_size, 1, num_points, 2, 2]
    rotations = rotations.expand(-1, time_out, -1, -1, -1)  # [batch_size, time_out, num_points, 2, 2]
    rotations = rotations.unsqueeze(0).expand(num_modes, -1, -1, -1, -1, -1)  # [num_modes, batch_size, time_out, num_points, 2, 2]

    # Reshape local_coords to [num_modes, time_out, batch_size, num_points, 1, 2]
    expanded_local_coords = local_coords.unsqueeze(-2)

    # Transpose the last two dimensions of local_coords to [num_modes, time_out, batch_size, num_points, 2, 1]
    expanded_local_coords = expanded_local_coords.transpose(-2, -1)

    # Rearrange dimensions to match rotations shape
    expanded_local_coords = expanded_local_coords.permute(0, 2, 1, 3, 4, 5)  # [num_modes, batch_size, time_out, num_points, 2, 1]

    # Perform the matrix multiplication
    rotated_coords = torch.matmul(rotations, expanded_local_coords).squeeze(-1)  # [num_modes, batch_size, time_out, num_points, 2]

    # Add the translation component
    translations_xy = translations[...,:2].unsqueeze(1).unsqueeze(1).expand(-1, num_modes, time_out, -1, -1)  # [batch_size, num_modes, time_out, num_points, 2]
    translations_xy = translations_xy.permute(1, 0, 2, 3, 4)  # [num_modes, batch_size, time_out, num_points, 2]
    global_coords = rotated_coords + translations_xy

    return global_coords.permute(0,2,1,3,4)
    


def l2_loss_fde(pred, data):
    fde_loss = torch.norm((pred[:, -1, :, :2].transpose(0, 1) - data[:, -1, :2].unsqueeze(1)), 2, dim=-1)
    ade_loss = torch.norm((pred[:, :, :, :2].transpose(1, 2) - data[:, :, :2].unsqueeze(0)), 2, dim=-1).mean(dim=2).transpose(0, 1)
    loss, min_inds = (fde_loss + ade_loss).min(dim=1)
    return 100.0 * loss.mean()



#pred[15T,16B,9N,6F]
def get_BVG_distributions_joint(pred):
    B = pred.size(0)
    T = pred.size(1)
    N = pred.size(2)
    mu_x = pred[..., 0].unsqueeze(-1)
    mu_y = pred[..., 1].unsqueeze(-1)
    sigma_x = pred[..., 2] # pred[:, :, :, 2]
    sigma_y = pred[..., 3]
    rho = pred[..., 4]

    #cov = torch.zeros((*rho.shape, 2, 2)).to(pred.device)
    cov = torch.zeros((*rho.shape, 2, 2)).to(pred.device)
    cov[..., 0, 0] = sigma_x ** 2
    cov[..., 1, 1] = sigma_y ** 2
    cov_val = rho * sigma_x * sigma_y
    cov[..., 0, 1] = cov_val
    cov[..., 1, 0] = cov_val

    biv_gauss_dist = MultivariateNormal(loc=torch.cat((mu_x, mu_y), dim=-1), covariance_matrix=cov)
    return biv_gauss_dist




def get_Laplace_dist_joint(pred):
    return Laplace(pred[..., :2], pred[..., 2:4])



# pred[15T,16B,9B,6F]
def nll_pytorch_dist_joint(pred, data, agents_masks):
    num_active_agents_per_timestep = agents_masks.sum(2) # [64,15]
    # biv_gauss_dist = get_BVG_distributions_joint(pred)


    biv_gauss_dist = get_Laplace_dist_joint(pred) # use :2 and 2:4 to build laplace
    loss = (((-biv_gauss_dist.log_prob(data).sum(-1) * agents_masks).sum(2)) / num_active_agents_per_timestep).sum(
        1)


    return loss





def nll_loss_multimodes_joint(pred, ego_data, agents_data, mode_probs, entropy_weight=1.0, kl_weight=1.0,
                              use_FDEADE_aux_loss=True, agent_types=None, predict_yaw=False,
                              args=None,
                              aux_output=None, model=None,
                              map_lanes=None,
                              post_train=False,
                              agents_in=None,
                              translations=None,
                              strategy=None):
    """
    Args:
      pred: [c, T, B, M, 5] torch.Size([6K, 15T, 64B, 9N, 6F])
      ego_data: [B, T, 5]  torch.Size([64, 15, 6])
      agents_data: [B, T, M, 5]  torch.Size([64, 15, 8, 6])
      mode_probs: [B, c], prior prob over modes [64,6]
      q, p if exist: T, B, N, K
      result_model: gmm for gmm model
      map_lanes [16,9,78,80,8] for lane error
      pred_P: [TBK,N,N]
      scene_dist: # [K,T,B,N,F]
      agents_in： list
        ego_in[B,T,11]
        agent_in [B,T,N,11]
        translations[B,N,3] include trans_x, trans_y, rot
    """
    gt_agents = torch.cat((ego_data.unsqueeze(2), agents_data), dim=2) # torch.Size([64B, 15T, 9N, 6K])
    modes = len(pred) # 6
    nSteps, batch_sz, N, pred_dim = pred[0].shape # 15, 64, 9, 6
    agents_masks = torch.cat((torch.ones(batch_sz, nSteps, 1).to(ego_data.device), agents_data[:, :, :, -1]), dim=-1) # [16B, 15T, 9N]


    loss = 0.0
    nll_loss = 0.0


    # compute posterior probability based on predicted prior and likelihood of predicted scene.
    log_lik = np.zeros((batch_sz, modes)) # [B,K]
    with torch.no_grad():
        for kk in range(modes):
            nll = nll_pytorch_dist_joint(pred[kk].transpose(0, 1), gt_agents[:, :, :, :2], agents_masks) # nll for every mode [16]
            log_lik[:, kk] = -nll.cpu().numpy() # nll [B] # 


    priors = mode_probs.detach().cpu().numpy()
    log_posterior_unnorm = log_lik + np.log(priors) #
    log_posterior = log_posterior_unnorm - np.expand_dims(special.logsumexp(log_posterior_unnorm, axis=1),axis=1)
    post_pr = np.exp(log_posterior) 
    post_pr = torch.tensor(post_pr).float().to(gt_agents.device)
    post_entropy = torch.mean(D.Categorical(post_pr).entropy()).item()


    for kk in range(modes):
        nll_k = nll_pytorch_dist_joint(pred[kk].transpose(0, 1), gt_agents[:, :, :, :2], agents_masks) * post_pr[:, kk] # 
        loss += nll_k.mean()
        nll_loss += nll_k.mean()


    # nll_loss = loss.clone()
    # Adding entropy loss term to ensure that individual predictions do not try to cover multiple modes.

    entropy_vals = []
    for kk in range(modes):
        entropy_vals.append(get_BVG_distributions_joint(pred[kk]).entropy()) 


        # [6K, 15T, 16B, 9N] -> [16B,6K,9N,15T] sum-> [B,6K,9N] mean-> [B,K] max->[B] ->mean[1]
        entropy_loss = torch.mean(torch.stack(entropy_vals).permute(2, 0, 3, 1).sum(3).mean(2).max(1)[0])
        loss += entropy_weight * entropy_loss


    # compute ADE/FDE loss - L2 norms with between best predictions and GT.
    if use_FDEADE_aux_loss:
        adefde_loss, post_prob_ade = l2_loss_fde_joint(pred, gt_agents, agents_masks, agent_types, predict_yaw)
    else:
        adefde_loss = torch.tensor(0.0).to(gt_agents.device)


    strategy_loss = torch.tensor(0.0).to(pred.device)


    #print("")
    if args.planning_post:
        gt_global = gt_agents[...,2:4] # [32bs,15tout,9n,2]
        pred_global = convert_pred_global(pred[...,:2],translations) # [6,15tout,32bs,9n,2]
        # strategy_loss [], best_traj_idx[8,9](gt),select_id
        strategy_loss, best_traj_idx, select_id, _ = cal_pl_loss(pred_global, gt_global, mode_probs, strategy)

    # KL divergence between the prior and the posterior distributions.
    kl_loss_fn = torch.nn.KLDivLoss(reduction='batchmean')  # type: ignore


    no_weight_kl_loss = kl_loss_fn(torch.log(mode_probs), post_pr)
    kl_loss = kl_weight * no_weight_kl_loss

    aux_loss = torch.tensor(0.0).to(pred.device)
    aux_loss = l2_loss_fde_proposal(aux_output, gt_agents,
                                        agents_masks, agent_types)


    return loss, kl_loss, post_entropy, adefde_loss, nll_loss, aux_loss, \
           entropy_loss, no_weight_kl_loss,\
           strategy_loss


def l2_loss_fde_joint(pred, data, agent_masks, agent_types, predict_yaw):
    """
    pred: 6k,15t,4b,9n,6f
    data: [32bs,15t,9n,6f]
    """
    fde_loss = (torch.norm((pred[:, -1, :, :, :2].transpose(0, 1) - data[:, -1, :, :2].unsqueeze(1)), 2, dim=-1) *
                agent_masks[:, -1:, :]).mean(-1) # [4b,6k,9n] 对n作min
    ade_loss = (torch.norm((pred[:, :, :, :, :2].transpose(1, 2) - data[:, :, :, :2].unsqueeze(0)), 2, dim=-1) *
                agent_masks.unsqueeze(0)).mean(-1).mean(dim=2).transpose(0, 1) # [6k,4b,15t,9n],nmin,15mean

    yaw_loss = torch.tensor(0.0).to(pred.device)
    if predict_yaw:
        vehicles_only = (agent_types[:, :, 0] == 1.0).unsqueeze(0)
        yaw_loss = torch.norm(pred[:, :, :, :, 5:].transpose(1, 2) - data[:, :, :, 4:5].unsqueeze(0), dim=-1).mean(2)  # across time
        yaw_loss = (yaw_loss * vehicles_only).mean(-1).transpose(0, 1)

    total_loss = (fde_loss + ade_loss + yaw_loss)
    loss, min_inds = total_loss.min(dim=1) # [4,6] min for 6
    post_prob = torch.zeros_like(total_loss) # [4,6]
    post_prob = post_prob.scatter(1, min_inds.unsqueeze(1), torch.ones(post_prob.shape[0], 1).to(post_prob.device))
    return 100.0 * loss.mean(), post_prob




def l2_loss_fde_proposal(pred, data, agent_masks, agent_types):
    """
    pred: 64B,15Tf,1+8N,3F,
    data:
    """
    fde_loss = (torch.norm((pred[:, -1, :, :, :2].transpose(0, 1) - data[:, -1, :, :2].unsqueeze(1)), 2, dim=-1) *
                agent_masks[:, -1:, :]).mean(-1) # [4b,6k,9n] 
    ade_loss = (torch.norm((pred[:, :, :, :, :2].transpose(1, 2) - data[:, :, :, :2].unsqueeze(0)), 2, dim=-1) *
                agent_masks.unsqueeze(0)).mean(-1).mean(dim=2).transpose(0, 1) # [6k,4b,15t,9n],

    yaw_loss = torch.tensor(0.0).to(pred.device)

    vehicles_only = (agent_types[:, :, 0] == 1.0).unsqueeze(0)
    yaw_loss = torch.norm(pred[:, :, :, :, 3:].transpose(1, 2) - data[:, :, :, 4:5].unsqueeze(0), dim=-1).mean(2)  # across time
    yaw_loss = (yaw_loss * vehicles_only).mean(-1).transpose(0, 1)

    loss, min_inds = (fde_loss + ade_loss + yaw_loss).min(dim=1) # [4,6] min for 6
    return 100.0 * loss.mean()
