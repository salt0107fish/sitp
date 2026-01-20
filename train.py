import os

import numpy as np
import random

from datasets.dataset import SITPDataset
from process_args import get_train_args

from models.SITP import SITP
import torch
import torch.distributions as D
from torch import optim, nn
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter

from utils.metric_helpers import min_xde_K
from utils.train_helpers import nll_loss_multimodes_joint

from tqdm import tqdm
import wandb


os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:768"


class Trainer:
    def __init__(self, args, results_dirname):
        self.args = args
        self.results_dirname = results_dirname
        random.seed(self.args.seed)
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        if torch.cuda.is_available() and not self.args.disable_cuda:
            self.device = torch.device("cuda")
            torch.cuda.manual_seed(self.args.seed)
        else:
            self.device = torch.device("cpu")

        self.initialize_dataloaders()
        self.initialize_model()
        self.optimiser = optim.Adam([
            {'params': self.model.parameters(),
            "lr": self.args.learning_rate,
            "eps": self.args.adam_epsilon},
        ])
        gamma=0.5


        self.optimiser_scheduler = MultiStepLR(self.optimiser, milestones=args.learning_rate_sched, gamma=gamma,
                                               verbose=True)

        if args.use_wandb is True:
            self.set_wandb(args)
        else:
            self.writer = SummaryWriter(log_dir=os.path.join(self.results_dirname, "tb_files"))
        self.smallest_minade_k = 5.0  # for computing best models
        self.smallest_minfde_k = 5.0  # for computing best models

        self.post_train = False

    def set_wandb(self, args):
        name = "sitp_train"
        wandb.init(
            project=name,
            resume="allow",
            config=args,
        )

    def initialize_dataloaders(self):
        train_dset = SITPDataset(dset_path=self.args.dataset_path, split_name="train",
                                        evaluation=False,
                                        args=self.args)
        val_dset = SITPDataset(dset_path=self.args.dataset_path, split_name="val",
                                        evaluation=False,
                                        args=self.args)

        self.num_other_agents = train_dset.num_others
        self.pred_horizon = train_dset.pred_horizon
        self.k_attr = train_dset.k_attr
        self.map_attr = train_dset.map_attr
        self.predict_yaw = train_dset.predict_yaw
        self.num_agent_types = train_dset.num_agent_types

        self.train_loader = torch.utils.data.DataLoader(
            train_dset, batch_size=self.args.batch_size, shuffle=True, num_workers=4, drop_last=False, pin_memory=False
        )
        self.val_loader = torch.utils.data.DataLoader(
            val_dset, batch_size=self.args.batch_size, shuffle=True, num_workers=4, drop_last=False, pin_memory=False
        )


        print("Train dataset loaded with length", len(train_dset))
        print("Val dataset loaded with length", len(val_dset))

    def initialize_model(self):
        self.model = SITP(  self.args,
                                    k_attr=self.k_attr,
                                    d_k=self.args.hidden_size,
                                    _M=self.num_other_agents,
                                    c=self.args.num_modes,
                                    T=self.pred_horizon,
                                    L_enc=self.args.num_encoder_layers,
                                    dropout=self.args.dropout,
                                    num_heads=self.args.tx_num_heads,
                                    L_dec=self.args.num_decoder_layers,
                                    tx_hidden_size=self.args.tx_hidden_size,
                                    use_map_lanes=self.args.use_map_lanes,
                                    num_agent_types=self.num_agent_types,
                                    predict_yaw=self.predict_yaw,).to(self.device)

    def _data_to_device(self, data, mode="train"):
        if mode == "train":
            ego_in, ego_out, agents_in, agents_out, context_img, agent_types, global_translations = data
            ego_in = ego_in.float().to(self.device)
            ego_out = ego_out.float().to(self.device)
            agents_in = agents_in.float().to(self.device)
            agents_out = agents_out.float().to(self.device)
            context_img = context_img.float().to(self.device)
            agent_types = agent_types.float().to(self.device)
            global_translations = global_translations.float().to(self.device)
            return ego_in, ego_out, agents_in, agents_out, context_img,agent_types, global_translations
        elif mode == "test":
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
        else:
            print("Error type in _data_to_device")


    def _compute_ego_errors(self, ego_preds, ego_gt):
        ego_gt = ego_gt.transpose(0, 1).unsqueeze(0)
        ade_losses = torch.mean(torch.norm(ego_preds[:, :, :, :2] - ego_gt[:, :, :, :2], 2, dim=-1), dim=1).transpose(0, 1).cpu().numpy()
        fde_losses = torch.norm(ego_preds[:, -1, :, :2] - ego_gt[:, -1, :, :2], 2, dim=-1).transpose(0, 1).cpu().numpy()
        return ade_losses, fde_losses

    def _compute_marginal_errors(self, preds, ego_gt, agents_gt, agents_in):
        agent_masks = torch.cat((torch.ones((len(agents_in), 1)).to(self.device), agents_in[:, -1, :, -1]), dim=-1).view(1, 1, len(agents_in), -1) # 1,1,4B,9N
        agent_masks[agent_masks == 0] = float('nan')
        agents_gt = torch.cat((ego_gt.unsqueeze(2), agents_gt), dim=2).unsqueeze(0).permute(0, 2, 1, 3, 4) # [1,15T,4B,9N,6F]
        error = torch.norm(preds[:, :, :, :, :2] - agents_gt[:, :, :, :, :2], 2, dim=-1) * agent_masks # [6,30,16,9]
        ade_losses = np.nanmean(error.cpu().numpy(), axis=1).transpose(1, 2, 0) # [16,9,6], 
        fde_losses = error[:, -1].cpu().numpy().transpose(1, 2, 0) # [16,9,6]
        return ade_losses, fde_losses

    def _compute_nll_errors(self, preds, ego_gt, agents_gt, agents_in):
        # preds[6K,15T,4B,9N,6F(x,y,sigx,sigy,rho,yaw)]
        agent_masks = torch.cat((torch.ones((len(agents_in), 1)).to(self.device), agents_in[:, -1, :, -1]), dim=-1).view(1, 1, len(agents_in), -1)# 1,1,4B,9N
        agent_masks[agent_masks == 0] = float('nan')
        agents_gt = torch.cat((ego_gt.unsqueeze(2), agents_gt), dim=2).unsqueeze(0).permute(0, 2, 1, 3, 4)# [1,15T,4B,9N,6F]

        mux = preds[..., 0]
        muy = preds[..., 1]
        sigmax = preds[..., 2]
        sigmay = preds[..., 3]
        gtx = agents_gt[..., 0]# [1,15,b,n]
        gty = agents_gt[..., 1]
        rho = preds[...,4] # [6,15,b,n]
        rho = torch.clamp(rho, min=-0.999, max=0.999)
        ohr = torch.sqrt(1 - rho ** 2) # [6,15,b,n]

        component_1 = torch.log(2 * np.pi * sigmax * sigmay * ohr)
        component_2 = (gtx - mux) ** 2 / sigmax ** 2 + \
                      (gty - muy) ** 2 / sigmay ** 2 - \
                      2 * rho * (gtx - mux) * (gty - muy) / (sigmax * sigmay)

        nll = component_1 + 0.5 * component_2 / (1 - rho ** 2)

        neg_log_prob = nll * agent_masks

        neg_log_prob = np.nanmean(neg_log_prob.cpu().numpy(), axis=1).transpose(1, 2, 0) # [16,9,6],

        return neg_log_prob # [16,9,6]



    def _compute_joint_errors(self, preds, ego_gt, agents_gt):
        agents_gt = torch.cat((ego_gt.unsqueeze(2), agents_gt), dim=2)
        agents_masks = agents_gt[:, :, :, -1]
        agents_masks[agents_masks == 0] = float('nan')
        ade_losses = []
        for k in range(self.args.num_modes):
            ade_error = (torch.norm(preds[k, :, :, :, :2].transpose(0, 1) - agents_gt[:, :, :, :2], 2, dim=-1)
                         * agents_masks).cpu().numpy()
            ade_error = np.nanmean(ade_error, axis=(1, 2))
            ade_losses.append(ade_error)
        ade_losses = np.array(ade_losses).transpose() # [16bs,6mode]

        fde_losses = []
        for k in range(self.args.num_modes):
            fde_error = (torch.norm(preds[k, -1, :, :, :2] - agents_gt[:, -1, :, :2], 2, dim=-1) * agents_masks[:, -1]).cpu().numpy()
            fde_error = np.nanmean(fde_error, axis=1)
            fde_losses.append(fde_error)
        fde_losses = np.array(fde_losses).transpose()

        return ade_losses, fde_losses

    def sitp_train(self):
        steps = 0
        epoch_num = self.args.num_epochs
        self.post_train = False 
        self.model.post_train = False

        if self.args.scene_post_path is not None:

            if os.path.exists(self.args.scene_post_path):
                weights_dict = torch.load(self.args.scene_post_path, map_location=self.device)
                load_weights_dict = {k: v for k, v in weights_dict["SIPT"].items()
                                     if self.model.state_dict()[k].numel() == v.numel()}
                self.model.load_state_dict(load_weights_dict, strict=False)
                print("Successful in loading weight for post scene nll")
            else:
                print("ERROR:Do not find scene_post_path but need to use!!!!!")

        for epoch in tqdm(range(0, epoch_num), desc="Train"):
            self.model.if_train = True

            if self.args.planning_post:
                for param in self.model.parameters():
                    param.requires_grad = False
                for param in self.model.strategy_decoder.parameters():
                    param.requires_grad = True
                for param in self.model.strategy_decoder.parameters():
                    param.requires_grad = True
                for param in self.model.strategy_map_attn.parameters():
                    param.requires_grad = True
                for param in self.model.strategy_predictor.parameters():
                    param.requires_grad = True
                self.model.S.requires_grad = True
                
                for param in self.model.parameters():
                    if param.requires_grad is False:
                        param.grad.data.zero_()
                for param in self.model.parameters():
                    if param.requires_grad is False:
                        param.grad = None

            print("Epoch:", epoch)
            epoch_marg_ade_losses = []
            epoch_marg_fde_losses = []
            epoch_marg_mode_probs = []
            epoch_mode_probs = []
            # for NLL
            epoch_nll_losses = []

            for i, data in enumerate(tqdm(self.train_loader)):
                # if i == 5:
                #     break # for debug
                ego_in, ego_out, agents_in, agents_out, map_lanes, agent_types, global_translations = self._data_to_device(data) # ego[32,5,11]-> [32,15,6] agent [32,5,8,11] -> [32,15,8,6] map[32,9,18,80,8], type[32,9,2]
                
                    
                pred_obs, mode_probs,pred_aux, strategy = self.model(ego_in, agents_in, map_lanes, agent_types)

                nll_loss, kl_loss, post_entropy, adefde_loss, \
                v_nll_loss, proposal_loss, entropy_loss,\
                    no_weight_kl_loss, \
                    planning_loss= \
                    nll_loss_multimodes_joint(pred_obs, ego_out, agents_out, mode_probs,
                                              entropy_weight=self.args.entropy_weight,
                                              kl_weight=self.args.kl_weight,
                                              use_FDEADE_aux_loss=self.args.use_FDEADE_aux_loss,
                                              agent_types=agent_types,
                                              predict_yaw=self.predict_yaw,
                                              args=self.args,
                                              aux_output=pred_aux,
                                              model=self.model,
                                              map_lanes=map_lanes,
                                              post_train=self.post_train,
                                              agents_in=[ego_in,agents_in],
                                              translations=global_translations,
                                              strategy=strategy)

                self.optimiser.zero_grad()



                (
                    self.args.strategy_weight * planning_loss + 
                    self.args.entropy_weight * entropy_loss +
                    v_nll_loss +
                    self.args.ade_weight * adefde_loss +
                    self.args.kl_weight * no_weight_kl_loss +
                    proposal_loss 
                    ).backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip_norm)
                self.optimiser.step()

                if self.args.use_wandb:
                    Loss_log ={
                        "Loss/nll": nll_loss.item(), # nll+me
                        "Loss/vnll": v_nll_loss.item(),
                        "Loss/me_nll": nll_loss.item()-v_nll_loss.item(), # me
                        "Loss/adefde": adefde_loss.item(),
                        "Loss/kl": kl_loss.item(),
                        "Loss/post_entropy": post_entropy,
                        "Loss/prior_entropy": torch.mean(D.Categorical(mode_probs).entropy()).item(),
                        "Loss/lr": self.optimiser_scheduler.get_lr()[-1],
                        "Loss/proposal": proposal_loss.item(),
                        "Loss/planning_loss": planning_loss.item(),
                        "Weight/ade_weight": self.args.ade_weight,
                        "Weight/entropy_weight": self.args.entropy_weight,
                        "Weight/kl_weight": self.args.kl_weight,
                        "Weight/joint_nll_weight": self.args.joint_nll_weight,
                        "Weight/scene_nll_weight": self.args.scene_nll_weight,
                    }
                    if self.args.multi_nll:
                        Loss_log["Loss/joint_nll_weight"] = self.learnable_weight[0].item()
                        Loss_log["Loss/lane_nll_weight"] = self.learnable_weight[1].item()
                    else:
                        Loss_log["Loss/joint_nll_weight"] = self.args.joint_nll_weight
                        Loss_log["Loss/lane_nll_weight"] = self.args.lane_nll_weight
                    wandb.log(Loss_log, step=steps)
                else:
                    self.writer.add_scalar("Loss/nll", nll_loss.item(), steps)
                    self.writer.add_scalar("Loss/adefde", adefde_loss.item(), steps)
                    self.writer.add_scalar("Loss/kl", kl_loss.item(), steps)

                with torch.no_grad():
                    ade_losses, fde_losses = self._compute_marginal_errors(pred_obs, ego_out, agents_out, agents_in)
                    epoch_marg_ade_losses.append(ade_losses.reshape(-1, self.args.num_modes))
                    epoch_marg_fde_losses.append(fde_losses.reshape(-1, self.args.num_modes))
                    epoch_marg_mode_probs.append(
                        mode_probs.unsqueeze(1).repeat(1, self.num_other_agents + 1, 1).detach().cpu().numpy().reshape(
                            -1, self.args.num_modes))

                    epoch_mode_probs.append(mode_probs.detach().cpu().numpy())


                    nll_losses = self._compute_nll_errors(pred_obs, ego_out, agents_out, agents_in) # [16B,9N,6K]
                    epoch_nll_losses.append(nll_losses.reshape(-1, self.args.num_modes))


                if i % 1 == 0:
                    print(i, "/", len(self.train_loader.dataset)//self.args.batch_size,
                            "Planning loss", round(planning_loss.item(), 2),
                            "NLL loss", round(nll_loss.item(), 2), "KL loss", round(kl_loss.item(), 2),
                            "Prior Entropy", round(torch.mean(D.Categorical(mode_probs).entropy()).item(), 2),
                            "Post Entropy", round(post_entropy, 2), "ADE+FDE loss", round(adefde_loss.item(), 2)
                            )

                steps += 1

            epoch_marg_ade_losses = np.concatenate(epoch_marg_ade_losses)
            epoch_marg_fde_losses = np.concatenate(epoch_marg_fde_losses)
            epoch_marg_mode_probs = np.concatenate(epoch_marg_mode_probs)
            mode_probs = np.concatenate(epoch_mode_probs)
            train_minade_c = min_xde_K(epoch_marg_ade_losses, epoch_marg_mode_probs, K=self.args.num_modes)
            train_minfde_c = min_xde_K(epoch_marg_fde_losses, epoch_marg_mode_probs, K=self.args.num_modes)

            # NLL
            epoch_nll_losses = np.concatenate(epoch_nll_losses)

            print("Train Marg. minADE c:", train_minade_c[0], "Train Marg. minFDE c:", train_minfde_c[0], "\n",

                  )

            # Log train metrics
            if self.args.use_wandb:
                Metric_log = {
                    "Metrics/Train/Marg. minADE {}".format(self.args.num_modes): train_minade_c[0],
                    "Metrics/Train/Marg. minFDE {}".format(self.args.num_modes): train_minfde_c[0],

                }
                wandb.log(Metric_log)
            else:
                self.writer.add_scalar("metrics/Train Marg. minADE {}".format(self.args.num_modes), train_minade_c[0], epoch)
                self.writer.add_scalar("metrics/Train Marg. minFDE {}".format(self.args.num_modes), train_minfde_c[0], epoch)


            self.optimiser_scheduler.step()
            self.save_model(epoch)
            self.sitp_evaluate(epoch, steps) 
            print("Best Scene minADE c", self.smallest_minade_k, "Best Scene minFDE c", self.smallest_minfde_k)


    def sitp_evaluate(self, epoch, steps):
        self.model.eval()
        self.model.if_train = False
        with torch.no_grad():
            val_marg_ade_losses = []
            val_marg_fde_losses = []
            val_marg_mode_probs = []
            val_scene_ade_losses = []
            val_scene_fde_losses = []
            val_mode_probs = []
            # for NLL
            val_nll_losses = []


            for i, data in enumerate(tqdm(self.val_loader, desc="Val")):
                ego_in, ego_out, agents_in, agents_out, context_img, agent_types, global_translations = self._data_to_device(data)
                pred_obs, mode_probs, pred_aux, strategy = self.model(ego_in, agents_in, context_img, agent_types)

                # Marginal metrics
                ade_losses, fde_losses = self._compute_marginal_errors(pred_obs, ego_out, agents_out, agents_in) # [16bs,9obj,6mode]
                val_marg_ade_losses.append(ade_losses.reshape(-1, self.args.num_modes))
                val_marg_fde_losses.append(fde_losses.reshape(-1, self.args.num_modes))

                val_marg_mode_probs.append(
                    mode_probs.unsqueeze(1).repeat(1, self.num_other_agents + 1, 1).detach().cpu().numpy().reshape(
                        -1, self.args.num_modes))

                # Joint metrics
                scene_ade_losses, scene_fde_losses = self._compute_joint_errors(pred_obs, ego_out, agents_out) # [16bs,6mode]
                val_scene_ade_losses.append(scene_ade_losses)
                val_scene_fde_losses.append(scene_fde_losses)
                val_mode_probs.append(mode_probs.detach().cpu().numpy())
                # NLL
                nll_losses = self._compute_nll_errors(pred_obs, ego_out, agents_out, agents_in)  # [16B,9N,6K]
                val_nll_losses.append(nll_losses)

            val_marg_ade_losses = np.concatenate(val_marg_ade_losses)
            val_marg_fde_losses = np.concatenate(val_marg_fde_losses)
            val_marg_mode_probs = np.concatenate(val_marg_mode_probs)

            val_scene_ade_losses = np.concatenate(val_scene_ade_losses)
            val_scene_fde_losses = np.concatenate(val_scene_fde_losses)
            val_mode_probs = np.concatenate(val_mode_probs)

            val_minade_c = min_xde_K(val_marg_ade_losses, val_marg_mode_probs, K=self.args.num_modes) #
            val_minfde_c = min_xde_K(val_marg_fde_losses, val_marg_mode_probs, K=self.args.num_modes)


            # NLL
            epoch_nll_losses = np.concatenate(val_nll_losses)




            # Log train metrics
            if self.args.use_wandb:
                Metric_log = {
                    "Metrics/Val/Marg. minADE {}".format(self.args.num_modes): val_minade_c[0],
                    "Metrics/Val/Marg. minFDE {}".format(self.args.num_modes): val_minfde_c[0],

                }
                wandb.log(Metric_log)
            else:
                self.writer.add_scalar("metrics/Val Marg. minADE {}".format(self.args.num_modes), val_minade_c[0], epoch)
                self.writer.add_scalar("metrics/Val Marg. minFDE {}".format(self.args.num_modes), val_minfde_c[0], epoch)


            print("Marg. minADE c:", val_minade_c[0], "Marg. minFDE c:", val_minfde_c[0])


            self.model.train()
            self.model.if_train = True

            self.save_model(minade_k=val_minade_c[0], minfde_k=val_minfde_c[0])


    def save_model(self, epoch=None, minade_k=None, minfde_k=None):
        if epoch is None:
            if minade_k < self.smallest_minade_k:
                self.smallest_minade_k = minade_k
                torch.save(
                    {
                        "SITP": self.model.state_dict(),
                        "optimiser": self.optimiser.state_dict(),
                    },
                    os.path.join(self.results_dirname, "best_models_ade.pth"),
                )

            if minfde_k < self.smallest_minfde_k:
                self.smallest_minfde_k = minfde_k
                torch.save(
                    {
                        "SITP": self.model.state_dict(),
                        "optimiser": self.optimiser.state_dict(),
                    },
                    os.path.join(self.results_dirname, "best_models_fde.pth"),
                )

        else:
            if self.args.debug:
                save_epoch = 1
            else:
                save_epoch = 40
            if epoch % save_epoch == 0 and epoch > 0:
                torch.save(
                    {
                        "SITP": self.model.state_dict(),
                        "optimiser": self.optimiser.state_dict(),
                    },
                    os.path.join(self.results_dirname, "models_%d.pth" % epoch),
                )

    def train(self):

        self.sitp_train()


if __name__ == "__main__":
    args, results_dirname = get_train_args()
    trainer = Trainer(args, results_dirname)
    trainer.train()
