import argparse
import json
import os
from collections import namedtuple



def get_train_args():
    parser = argparse.ArgumentParser(description="SITP")
    # Section: General Configuration
    parser.add_argument("--exp-id", type=str, default=None, help="Experiment identifier")
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument("--disable-cuda", action="store_true", help="Disable CUDA")
    parser.add_argument("--save-dir", type=str, default=".", help="Directory for saving results")

    # Section: Dataset
    parser.add_argument("--dataset", type=str, default="interaction-dataset",help="Dataset to train on.")
    parser.add_argument("--dataset-path", type=str, default="./datasets/interaction_dataset/interaction_dataset_h5_file/", help="Path to dataset files.")
    parser.add_argument("--use-map-lanes", type=bool, default=True, help="Use map lanes if applicable.")


    parser.add_argument("--num-modes", type=int, default=6)
    parser.add_argument("--hidden-size", type=int, default=128, help="Model's hidden size.")
    parser.add_argument("--num-encoder-layers", type=int, default=2)
    parser.add_argument("--num-decoder-layers", type=int, default=2)
    parser.add_argument("--tx-hidden-size", type=int, default=384,
                        help="hidden size of transformer layers' feedforward network.")
    parser.add_argument("--tx-num-heads", type=int, default=16, help="Transformer number of heads.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout strenght used throughout model.")
    parser.add_argument("--hist-len", type=int, default=10)
    parser.add_argument("--pred-len", type=int, default=30)
    parser.add_argument("--debug", type=bool, default=False)


    # Section: Loss Function
    parser.add_argument("--entropy-weight", type=float, default=40.0, metavar="lamda", help="Weight of entropy loss.")
    parser.add_argument("--kl-weight", type=float, default=20.0, metavar="lamda", help="Weight of entropy loss.")
    parser.add_argument("--ade-weight", type=float, default=1.0, metavar="lamda", help="Weight of entropy loss.")
    parser.add_argument("--use-FDEADE-aux-loss", type=bool, default=True,
                        help="Whether to use FDE/ADE auxiliary loss in addition to NLL (accelerates learning).")


    # Section: Training params:
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--adam-epsilon", type=float, default=1e-4, help="Adam optimiser epsilon value")
    parser.add_argument("--learning-rate-sched", type=int, nargs='+', default=[10, 20, 30, 40, 50],
                        help="Learning rate Schedule.")
    parser.add_argument("--grad-clip-norm", type=float, default=5, metavar="C", help="Gradient clipping norm")
    parser.add_argument("--num-epochs", type=int, default=150, metavar="I", help="number of iterations through the dataset.")

    parser.add_argument("--use-interp", type=bool, default=True)
    parser.add_argument("--use-wandb", type=bool, default=False,
                        help="Use wandb, else tensorflow.")


    # parser.add_argument("--output-model", type=str, default="LN", help="choose from [LN, GRU].")
    # parser.add_argument("--outmodel", type=str, default="mlp")
    # parser.add_argument("--out-mode", type=str, default=None, help="choose from [v, None].")
    # parser.add_argument("--cvae-model", type=str, default="z_mode", help="choose from [z_mode, qzxy, proposal, qzxyv2, qzxyv3].")


    parser.add_argument("--scene-post-path", type=str,
                        default=None)


    parser.add_argument("--strategy", type=bool, default=False) # 
    parser.add_argument("--strategy-weight", type=float, default=1)
    parser.add_argument("--planning-post", type=bool, default=False) #


    args = parser.parse_args()


    print("===============================")
    print("Hist len: " + str(args.hist_len))
    print("Pred len: " + str(args.pred_len))

    print("Use wandb: " + str(args.use_wandb))

    # print("Use out/uncertain model: " + str(args.outmodel))

    print("Using planning loss: " + str(args.strategy) + " with weight " + str(args.strategy_weight))

    print("==============END==============")

    results_dirname = create_results_folder(args)
    save_config(args, results_dirname)


    return args, results_dirname


def get_eval_args():
    parser = argparse.ArgumentParser(description="SITP")
    parser.add_argument("--dataset-path", type=str, default="./datasets/interaction_dataset/interaction_dataset_h5_file", help="Load model checkpoint")
    parser.add_argument("--models-path", type=str, default="pretrained/use_model_strategy.pth", help="Dataset path.")

    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--disable-cuda", action="store_true", help="Disable CUDA")
    parser.add_argument("--hist-len", type=int, default=10)
    parser.add_argument("--pred-len", type=int, default=30)

    parser.add_argument("--use-interp", type=bool, default=True)

    parser.add_argument("--strategy", type=bool, default=False)


    args = parser.parse_args()

    config, model_dirname = load_config(args.models_path)
    config = namedtuple("config", config.keys())(*config.values())
    return args, config, model_dirname


def create_results_folder(args):
    model_configname = ""
    model_configname += "Sind" if "sind" in args.dataset_path else "Interaction" # 
    model_configname += "_wandb" if args.use_wandb is True else ""
    if args.exp_id is not None:
        model_configname += ("_" + args.exp_id)
    model_configname += "_s"+str(args.seed)

    result_dirname = os.path.join(args.save_dir, "results", args.dataset, model_configname)
    if os.path.isdir(result_dirname) and args.use_wandb:
        answer = input(result_dirname + " exists. \n Do you wish to overwrite? (y/n)")
        if 'y' in answer:
            if os.path.isdir(os.path.join(result_dirname, "tb_files")):
                for f in os.listdir(os.path.join(result_dirname, "tb_files")):
                    os.remove(os.path.join(result_dirname, "tb_files", f))
        else:
            exit()
    os.makedirs(result_dirname, exist_ok=True)
    return result_dirname


def save_config(args, results_dirname):
    argparse_dict = vars(args)
    with open(os.path.join(results_dirname, 'config.json'), 'w') as fp:
        json.dump(argparse_dict, fp)


def load_config(model_path):
    # model_dirname = "/" + os.path.join(*model_path.split("/")[:-1])
    model_dirname = os.path.join(*model_path.split("/")[:-1])
    # assert os.path.isdir(model_dirname)
    with open(os.path.join(model_dirname, 'config.json'), 'r') as fp:
        config = json.load(fp)
    return config, model_dirname
