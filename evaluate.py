import random
import numpy as np
import torch
from datasets.dataset import SITPDataset
from models.SITP import SITP
from utils.metric_helpers import min_xde_K, yaw_from_predictions, interpolate_trajectories, translate_agents

import argparse
import json
from tqdm import tqdm
import os
from collections import namedtuple

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:768"

def process_args():
    parser = argparse.ArgumentParser(description="SITP")
    parser.add_argument("--dataset-path", type=str, default="./datasets/interaction_dataset/interaction_dataset_h5_file", help="dataset path")
    parser.add_argument("--models-path", type=str, default="pretrained/new.pth", help="model path.")

    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--disable-cuda", action="store_true")
    parser.add_argument("--hist-len", type=int, default=10)
    parser.add_argument("--pred-len", type=int, default=30)
    parser.add_argument("--strategy", type=bool, default=True)

    args = parser.parse_args()

    model_dirname = os.path.join(*args.models_path.split("/")[:-1])
    with open(os.path.join(model_dirname, 'config.json'), 'r') as fp:
        config = json.load(fp)
    config = namedtuple("config", config.keys())(*config.values())

    return args, config, model_dirname


class Evaluator:
    def __init__(self, args, model_config, model_dirname):
        self.args = args
        self.model_config = model_config
        self.model_dirname = model_dirname

        random.seed(self.model_config.seed)
        np.random.seed(self.model_config.seed)
        torch.manual_seed(self.model_config.seed)

        if torch.cuda.is_available() and not self.args.disable_cuda:
            self.device = torch.device("cuda")
            torch.cuda.manual_seed(self.model_config.seed)
        else:
            self.device = torch.device("cpu")

        self.build_dataloader()
        self.build_model()


    def build_dataloader(self):
        evaluate_dataset = SITPDataset(dset_path=self.args.dataset_path, 
                                       split_name="val",
                                       evaluation=True, 
                                        args=self.args,) 
        self.num_other_agents = evaluate_dataset.num_others
        self.pred_horizon = evaluate_dataset.pred_horizon
        self.k_attr = evaluate_dataset.k_attr
        self.map_attr = evaluate_dataset.map_attr
        self.predict_yaw = evaluate_dataset.predict_yaw
        self.num_agent_types = evaluate_dataset.num_agent_types

        self.val_loader = torch.utils.data.DataLoader(
            evaluate_dataset, batch_size=self.args.batch_size, shuffle=False, num_workers=12, drop_last=False,
            pin_memory=False
        )

    def build_model(self):
        self.model = SITP(args,
                        k_attr=self.k_attr,
                        d_k=self.model_config.hidden_size,
                        _M=self.num_other_agents,
                        c=self.model_config.num_modes,
                        T=self.pred_horizon,
                        L_enc=self.model_config.num_encoder_layers,
                        dropout=self.model_config.dropout,
                        num_heads=self.model_config.tx_num_heads,
                        L_dec=self.model_config.num_decoder_layers,
                        tx_hidden_size=self.model_config.tx_hidden_size,
                        use_map_lanes=self.model_config.use_map_lanes,
                        map_attr=self.map_attr,
                        num_agent_types=self.num_agent_types,
                        predict_yaw=self.predict_yaw).to(self.device)

        model_dicts = torch.load(self.args.models_path, map_location=self.device)
        self.model.load_state_dict(model_dicts["SITP"])
        self.model.eval()

        model_parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        num_params = sum([np.prod(p.size()) for p in model_parameters])
        print("Number of Model Parameters:", num_params)

    def _data_to_device(self, data):

        ego_in, ego_out, agents_in, agents_out, context_img, agent_types, ego_out_model, agents_out_model = data
        ego_in = ego_in.float().to(self.device)
        ego_out = ego_out.float().to(self.device)
        agents_in = agents_in.float().to(self.device)
        agents_out = agents_out.float().to(self.device)
        context_img = context_img.float().to(self.device)
        agent_types = agent_types.float().to(self.device)
        ego_out_model = ego_out_model.float().to(self.device)
        agents_out_model = agents_out_model.float().to(self.device)
        return ego_in, ego_out, agents_in, agents_out, context_img, agent_types, ego_out_model, agents_out_model

    def postprocess(self,pred_obs,orig_ego_in,orig_agents_in,agent_types,translations):
        pred_obs = interpolate_trajectories(pred_obs)
        pred_obs = yaw_from_predictions(pred_obs, orig_ego_in, orig_agents_in)
        _, pred_obs, _ = translate_agents(pred_obs.cpu().numpy(),
                                        agent_types.cpu().numpy(),
                                        orig_ego_in.cpu().numpy(),
                                        orig_agents_in.cpu().numpy(),
                                        translations.cpu().numpy(),
                                        device=self.device,
                                        args=self.args)
        
        return pred_obs

    def _compute_marginal_errors(self, preds, ego_gt, agents_gt, agents_in):
        agent_masks = torch.cat((torch.ones((len(agents_in), 1)).to(self.device), agents_in[:, -1, :, -1]), dim=-1).view(1, 1, len(agents_in), -1)
        agent_masks[agent_masks == 0] = float('nan')
        agents_gt = torch.cat((ego_gt.unsqueeze(2), agents_gt), dim=2).unsqueeze(0).permute(0, 2, 1, 3, 4)
        error = torch.norm(preds[:, :, :, :, :2] - agents_gt[:, :, :, :, :2], 2, dim=-1) * agent_masks 
        ade_losses = np.nanmean(error.cpu().numpy(), axis=1).transpose(1, 2, 0)
        fde_losses = error[:, -1].cpu().numpy().transpose(1, 2, 0)
        return ade_losses, fde_losses


    def evaluate(self):
        self.model.eval()
        self.model.if_train = False
        self.model._M = self.num_other_agents 

        with torch.no_grad():

            ade_losses_list = []
            fde_losses_list = []
            mode_probs_list = []

            for i, data in enumerate(tqdm(self.val_loader)):

                orig_ego_in, orig_agents_in, original_roads, translations, global_translations = data[8:]

                data = data[:8]
                orig_ego_in = orig_ego_in.float().to(self.device)
                orig_agents_in = orig_agents_in.float().to(self.device)

                ego_in, ego_out, agents_in, agents_out, context_img, agent_types, ego_out_model, agents_out_model = self._data_to_device(
                    data)

                pred_obs, mode_probs, pred_aux, strategy = self.model(ego_in, agents_in,
                                                          context_img, agent_types)

                pred_obs = self.postprocess(pred_obs,orig_ego_in,orig_agents_in,agent_types,translations)
                
                # Marginal metrics
                ade_losses, fde_losses = self._compute_marginal_errors(pred_obs, ego_out, agents_out, agents_in)
                ade_losses_list.append(ade_losses.reshape(-1, self.model_config.num_modes))
                fde_losses_list.append(fde_losses.reshape(-1, self.model_config.num_modes))
                mode_probs_list.append(
                    mode_probs.unsqueeze(1).repeat(1, self.num_other_agents + 1, 1).detach().cpu().numpy().reshape(
                        -1, self.model_config.num_modes)) 
                

            ade_losses = np.concatenate(ade_losses_list)
            fde_losses = np.concatenate(fde_losses_list)
            mode_probs = np.concatenate(mode_probs_list)

            val_minade_c = min_xde_K(ade_losses, mode_probs, K=self.model_config.num_modes)
            val_minfde_c = min_xde_K(fde_losses, mode_probs, K=self.model_config.num_modes)

            print("The results is: ")
            print("minADE:", val_minade_c[0], "minFDE:", val_minfde_c[0])

if __name__ == '__main__':
    args, config, model_dirname = process_args()
    evaluator = Evaluator(args, config, model_dirname)
    evaluator.evaluate()
