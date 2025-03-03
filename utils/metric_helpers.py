# -*- coding: utf-8 -*-
"""
This script is originally from Autobot.
URL: https://github.com/roggirg/AutoBots

Modifications made by salt0107fish.
"""

import numpy as np
import torch


def min_xde_K(xdes, probs, K):
    best_ks = probs.argsort(axis=1)[:, -K:] 
    dummy_rows = np.expand_dims(np.arange(len(xdes)), 1)
    new_xdes = xdes[dummy_rows, best_ks] 
    return np.nanmean(np.sort(new_xdes), axis=0)

def yaw_from_predictions(preds, ego_in, agents_in):
    device = preds.device
    new_preds = preds.clone()
    last_obs_yaws = torch.cat((ego_in[:, -1, 7].unsqueeze(-1), agents_in[:, -1, :, 7]), dim=-1)
    T = preds.shape[1]
    yaws = torch.zeros((preds.shape[:4])).to(device)
    for t in range(T):
        yaws[:, t] = last_obs_yaws.unsqueeze(0) + preds[:, t, :, :, 2]
    new_preds[:, :, :, :, 2] = yaws

    return new_preds


def interpolate_trajectories(preds):
    device = preds.device
    K = preds.shape[0]
    T_in = preds.shape[1]
    B = preds.shape[2]
    N = preds.shape[3]
    out = preds.shape[4]

    new_preds = torch.zeros((K, 2*T_in, B, N, out)).to(device) # [K,2T,B,N,6]
    T_in = preds.shape[1]
    preds = preds.permute(0, 2, 3, 1, 4).reshape(-1, T_in, out) # [-1,T,6]
    new_idx = 0
    for t in range(T_in):
        if t == 0:
            new_pred = (preds[:, t, :2] - torch.tensor([[0.0, 0.0]]).to(device)) / 2.0
            new_pred = new_pred.view(K, B, N, 2)
            pred_t = preds[:, t, :2].view(K, B, N, 2)
            new_pred_yaw = (preds[:, t, -1] - torch.tensor([0.0]).to(device)) / 2.0
            new_pred_yaw = new_pred_yaw.view(K, B, N)
            pred_t_yaw = preds[:, t, -1].view(K, B, N)
            new_sigma_corr = (preds[:, t, 2:5] - torch.tensor([[0.0, 0.0, 0.0]]).to(device)) / 2.0
            new_sigma_corr = new_sigma_corr.view(K, B, N, 3)
            sigma_corr_t = preds[:, t, 2:5].view(K, B, N, 3)

            new_preds[:, new_idx, :, :, :2] = new_pred
            new_preds[:, new_idx, :, :, 2] = new_pred_yaw
            new_preds[:, new_idx, :, :, 3:6] = new_sigma_corr
            new_idx += 1
            new_preds[:, new_idx, :, :, :2] = pred_t
            new_preds[:, new_idx, :, :, 2] = pred_t_yaw
            new_preds[:, new_idx, :, :, 3:6] = sigma_corr_t
            new_idx += 1
        else:
            new_pred = ((preds[:, t, :2] - preds[:, t-1, :2]) / 2.0) + preds[:, t-1, :2]
            new_pred = new_pred.view(K, B, N, 2)
            pred_t = preds[:, t, :2].view(K, B, N, 2)
            new_pred_yaw = ((preds[:, t, -1] - preds[:, t-1, -1]) / 2.0) + preds[:, t-1, -1]
            new_pred_yaw = new_pred_yaw.view(K, B, N)
            pred_t_yaw = preds[:, t, -1].view(K, B, N)
            new_sigma_corr = ((preds[:, t, 2:5] - preds[:, t-1, 2:5]) / 2.0) + preds[:, t-1, 2:5]
            new_sigma_corr = new_sigma_corr.view(K, B, N, 3)
            sigma_corr_t = preds[:, t, 2:5].view(K, B, N, 3)

            new_preds[:, new_idx, :, :, :2] = new_pred
            new_preds[:, new_idx, :, :, 2] = new_pred_yaw
            new_preds[:, new_idx, :, :, 3:6] = new_sigma_corr
            new_idx += 1
            new_preds[:, new_idx, :, :, :2] = pred_t
            new_preds[:, new_idx, :, :, 2] = pred_t_yaw
            new_preds[:, new_idx, :, :, 3:6] = sigma_corr_t
            new_idx += 1

    return new_preds


def make_2d_rotation_matrix(angle_in_radians: float) -> np.ndarray:
    return np.array([[np.cos(angle_in_radians), -np.sin(angle_in_radians)],
                     [np.sin(angle_in_radians), np.cos(angle_in_radians)]])


def convert_local_coords_to_global(coordinates: np.ndarray, yaw: float) -> np.ndarray:
    transform = make_2d_rotation_matrix(angle_in_radians=-yaw)
    if len(coordinates.shape) > 2:
        coord_shape = coordinates.shape
        return np.dot(transform, coordinates.reshape((-1, 2)).T).T.reshape(*coord_shape)
    return np.dot(transform, coordinates.T).T[:, :2]

def convert_local_cov_to_global(covs: np.ndarray, yaw: float) -> np.ndarray:

    transform = make_2d_rotation_matrix(angle_in_radians=-yaw)  

    sigma_x = covs[..., 0]
    sigma_y = covs[..., 1]
    rho = covs[..., 2]

    sigma_cov = np.array(
        [
            [sigma_x ** 2, rho * sigma_x * sigma_y],
            [rho * sigma_x * sigma_y, sigma_y ** 2],
        ]
    )

    sigma_cov = np.transpose(sigma_cov, axes=[2,3,0,1]) 

    sigma_cov_trans = transform @ sigma_cov @ transform.T 

    new_cov = np.zeros_like(covs)
    new_cov[..., 0] = np.sqrt(sigma_cov_trans[...,0,0])
    new_cov[..., 1] = np.sqrt(sigma_cov_trans[..., 1,1])
    new_cov[..., 2] = sigma_cov_trans[..., 0, 1] /(new_cov[..., 0]*new_cov[..., 1])

    return new_cov


def return_circle_list(x, y, l, w, yaw):

    r = w/np.sqrt(2)
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    if l < 4.0:
        c1 = [x-(l-w)/2*cos_yaw, y-(l-w)/2*sin_yaw]
        c2 = [x+(l-w)/2*cos_yaw, y+(l-w)/2*sin_yaw]
        c = [c1, c2]
    elif l >= 4.0 and l < 8.0:
        c0 = [x, y]
        c1 = [x-(l-w)/2*cos_yaw, y-(l-w)/2*sin_yaw]
        c2 = [x+(l-w)/2*cos_yaw, y+(l-w)/2*sin_yaw]
        c = [c0, c1, c2]
    else:
        c0 = [x, y]
        c1 = [x-(l-w)/2*cos_yaw, y-(l-w)/2*sin_yaw]
        c2 = [x+(l-w)/2*cos_yaw, y+(l-w)/2*sin_yaw]
        c3 = [x-(l-w)/2*cos_yaw/2, y-(l-w)/2*sin_yaw/2]
        c4 = [x+(l-w)/2*cos_yaw/2, y+(l-w)/2*sin_yaw/2]
        c = [c0, c1, c2, c3, c4]
    for i in range(len(c)):
        c[i] = np.stack(c[i], axis=-1)
    c = np.stack(c, axis=-2)
    return c


def return_collision_threshold(w1, w2):
    return (w1 + w2) / np.sqrt(3.8)


def translate_agents(preds, agent_types, ego_in, agents_in, translations,
                                 device="cpu", args=None):
    
    new_preds = preds.copy()
    last_obs_yaws = np.concatenate((ego_in[:, -1, 7:8], agents_in[:, -1, :, 7]), axis=-1)
    angles_of_rotation = (np.pi / 2) + np.sign(-last_obs_yaws) * np.abs(last_obs_yaws)
    lengths = np.concatenate((ego_in[:, -1, 8:9], agents_in[:, -1, :, 8]), axis=-1)
    widths = np.concatenate((ego_in[:, -1, 9:10], agents_in[:, -1, :, 9]), axis=-1)

    vehicles_only = agent_types[:, :, 0] == 1.0 
    K = preds.shape[0]
    N = preds.shape[3]
    B = ego_in.shape[0]
    batch_collisions = np.zeros(B)
    for b in range(B):
        agents_circles = []
        agents_widths = []
        curr_preds = preds[:, :, b]
        for n in range(N):
            if not vehicles_only[b, n]: #
                if agent_types[b, n, 1]: #
                    if n == 0:
                        diff = ego_in[b, -1, 0:2] - ego_in[b, 0, 0:2]
                        yaw = np.arctan2(diff[1], diff[0])
                    else:
                        diff = agents_in[b, -1, n - 1, 0:2] - agents_in[b, 0, n - 1, 0:2]
                        yaw = np.arctan2(diff[1], diff[0])
                    angle_of_rotation = (np.pi / 2) + np.sign(-yaw) * np.abs(yaw)
                    new_preds[:, :, b, n, :2] = convert_local_coords_to_global(coordinates=curr_preds[:, :, n, :2],
                                                                               yaw=angle_of_rotation) + \
                                                translations[b, n].reshape(1, 1, -1)
                    new_preds[:, :, b, n, 3:6] = convert_local_cov_to_global(covs=curr_preds[:, :, n, 3:6], yaw=angle_of_rotation)

                continue
            yaw = angles_of_rotation[b, n]
            new_preds[:, :, b, n, :2] = convert_local_coords_to_global(coordinates=curr_preds[:, :, n, :2], yaw=yaw) + \
                                        translations[b, n].reshape(1, 1, -1)
            new_preds[:, :, b, n, 3:6] = convert_local_cov_to_global(covs=curr_preds[:, :, n, 3:6], yaw=yaw)

            preds_for_circle = new_preds[:, :, b, n, :3].transpose(1, 0, 2)
            circles_list = return_circle_list(x=preds_for_circle[:, :, 0],
                                              y=preds_for_circle[:, :, 1],
                                              l=lengths[b, n], w=widths[b, n],
                                              yaw=preds_for_circle[:, :, 2])
            agents_circles.append(circles_list)
            agents_widths.append(widths[b, n])
            
        collisions = np.zeros(K)
        for n in range(len(agents_circles)):
            curr_agent_circles = agents_circles[n]
            for _n in range(len(agents_circles)):
                if n == _n:
                    continue
                other_agent_circles = agents_circles[_n]
                threshold_between_agents = return_collision_threshold(agents_widths[n], agents_widths[_n])

                for curr_circle_idx in range(curr_agent_circles.shape[2]):
                    for other_circle_idx in range(other_agent_circles.shape[2]):
                        dists = np.linalg.norm(curr_agent_circles[:, :, curr_circle_idx] - other_agent_circles[:, :, other_circle_idx], axis=-1)
                        collisions += (dists <= threshold_between_agents).sum(0)

        batch_collisions[b] = (collisions > 0.0).sum() / K

    return batch_collisions, torch.from_numpy(new_preds).to(device), vehicles_only

