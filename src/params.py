#params.py
import argparse
from distutils import util
import yaml
import sys

def parse_args():
    parser = argparse.ArgumentParser(description='MVP')

    #DDP
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--local_world_size", type=int, default=1)
    ## Basics
    parser.add_argument("--config_file", help="Configuration file containing parameters", type=str)
    parser.add_argument("--mode", help="Train/Attack", type=str, default = "train", choices = ["train","attack","eval"])
    parser.add_argument("--dataset", help="Select dataset name", type=str, default = "imdb", choices = ["ag_news", "boolq", "sst2"])
    parser.add_argument("--train_size", help = "fraction of training examples to use", type = float, default = 0.95)
    parser.add_argument("--val_size", help = "fraction of val examples to use", type = float, default = 0.05)
    parser.add_argument("--model", help="Model Architecture", type=str, default = "bert-base-uncased", choices = ["bert-base-uncased" ,"roberta-base", "gpt2"])

    parser.add_argument("--model_id", help = "For Saving", type = str, default = '0')
    parser.add_argument("--model_type", help = "Which model to use", choices = "[mvp, untrained_mvp, mlp_ft, untrained_mlp_ft, projectcls, clsprompt, lpft_sparse, lpft_dense]", type = str, default = 'mvp')
    parser.add_argument("--seed", help = "Seed", type = int, default = 0)
    parser.add_argument("--checkpoint_interval", help = "Save model after every N steps", type = int, default = 1000)
    
    # MVP specific params
    parser.add_argument("--pool_label_words", help = "How to pool the logits of label words?", type = str, default = "max", choices = ["max", "mean"])
    parser.add_argument("--pool_templates", help = "How to pool the logits of templates?", type = str, default = "mean", choices = ["max", "mean"])
    parser.add_argument("--verbalizer_file", help = "Path for verbalizer file", type = str, default = None)
    parser.add_argument("--template_file", help = "Path for template file", type = str, default = None)
    parser.add_argument("--num_template", help = "which template to use in the file (-1 for all templates)", type = int, default = -1)

    #HPARAMS
    parser.add_argument("--num_epochs", help = "Number of Epochs", type = int, default = 20)
    parser.add_argument("--patience", help = "Number of Epochs", type = int, default = 10)
    parser.add_argument("--batch_size", help = "Batch Size for Train Set (Default = 8)", type = int, default = 8)
    parser.add_argument("--lr", help = "Learning Rate", type = float, default = 1e-5)
    parser.add_argument("--weight_decay", help = "Weight Decay", type = float, default = 0.01)
    parser.add_argument("--max_length", help = "Max Sequence Length", type = int, default = 512)
    
    #TEST
    parser.add_argument("--path", help = "Path for test model load", type = str, default = "None")
    parser.add_argument("--attack_name", help = "Attack Name", type = str, default = "textfooler", choices = ["none", "textfooler", "textbugger"])
    parser.add_argument("--num_examples", help = "number of test examples", type = int, default = 1000)
    parser.add_argument("--query_budget", help = "Query Budget per example (-1 for no budget)", type = int, default = -1)
    parser.add_argument("--split", help = "split to attack on", type = str, default = "test", choices = ["train", "validation", "test"])
    

    #Lp Norm Dependent
    parser.add_argument("--alpha", help = "Step Size for L2 attacks", type = float, default = None)
    parser.add_argument("--num_iter", help = "PGD iterations", type = int, default = 1)
    parser.add_argument("--epsilon", help = "Epsilon Radius PGD attacks", type = float, default = 1.0)
    parser.add_argument("--norm", help = "norm to use for adversarial augmentation", type = str, default = "l2", choices = ["l2","linf"])
    parser.add_argument("--adv_augment", help = "Use adversarial training or not", type = int, default = 0, choices = [0,1])

    return parser

def add_config(args):
    data = yaml.load(open(args.config_file,'r'))
    args_dict = args.__dict__
    for key, value in data.items():
        if('--'+key in sys.argv and args_dict[key] != None): ## Giving higher priority to arguments passed in cli
            continue
        if isinstance(value, list):
            args_dict[key] = []
            args_dict[key].extend(value)
        else:
            args_dict[key] = value
    return args