#main.py
from __future__ import absolute_import


# import random, src.params as params, os
import random
import src.params as params
import os
import numpy as np
import torch
#test

def set_seeds(seed, reproduce = False):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    if reproduce:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        

if __name__ == "__main__":
    
    # Load Params from CLI / Config File
    parser = params.parse_args()
    args = parser.parse_args()
    args = params.add_config(args) if args.config_file != None else args

    
    num_labels_map = {"imdb":2, "sst2":2, "amazon":5, "snli":3,"ag_news":4,"rotten_tomatoes":2,"emotion":6, "emotion2":2, "boolq":2, "movie_rationales":2}
    args.num_labels = num_labels_map[args.dataset]
    set_seeds(args.seed)


    print(args)

    # Model Saving and Logging Directory
    print(args.model)
    root = f"./checkpoints/{args.dataset}/{args.model}"

    model_dir = f"{root}/model_{args.model_id}"
    
    if args.mode == "attack":
        model_dir = f"{args.path}attack_logs/"
    args.model_dir = model_dir
    args.cache_dir = args.model_dir + "/cache"
    if(not os.path.exists(model_dir)):
        os.makedirs(model_dir)
    print("Model Directory:", args.model_dir)
       
    with open(f"{model_dir}/model_info.txt", "w") as f:
        from json import dump
        dump(args.__dict__, f, indent=2)
    
    if args.mode == "train":
        from src.train import trainer
        trainer(args)
    
    elif args.mode == "attack":
        from src.test import attacker
        attacker(args)
    