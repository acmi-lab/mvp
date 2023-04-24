from __future__ import absolute_import
import sys, os
# sys.path.append("textattack_lib/textattack/.")
sys.path.append("customattacks/.")
from textattack.attacker import Attacker
from TextFoolerCustom import TextFoolerCustom
from TextBuggerCustom import TextBuggerCustom
from src.utils.funcs import *
from src.models import get_model
from textattack import AttackArgs
from textattack.datasets import HuggingFaceDataset

import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

# CUDA_VISIBLE_DEVICES=1 python main.py --mode attack --path models/imdb/bert-base-uncased/model_unk_rand_0.25_no-fill_limit_lm/final_model/ --attack_name pruthi --swap_size 10000 --infill no-fill

def ood_evaluation_loop(args, model):
    ood_datasets = {"movie_rationales":[], "rotten_tomatoes":[],"emotion2":[],"amazon_polarity":[]}
    model.mode = "eval"
    for dataset in ood_datasets.keys():
        args.dataset = dataset
        eval_results = evaluate_model(model, args)
        ood_datasets[dataset] = eval_results
        print(eval_results)
    import json

    with open(f'{args.model_dir}/ood.json', 'w') as fp:
        json.dump(ood_datasets, fp)



def attacker(args):
    #ipdb.set_trace()
    if not os.path.exists(args.model_dir): os.makedirs(args.model_dir)
    file = open(f"{args.model_dir}/eval.txt", "a")  
    def myprint(a): print(a); file.write(a); file.write("\n"); file.flush()
    chkpt_name = os.path.basename(args.path)
    #train dataset is needed to get the right vocabulary for the problem
    my_dataset, tokenizer, data_collator = prepare_huggingface_dataset(args)
    verbalizer, templates = get_prompts(args) 
    model = get_model(args, my_dataset, tokenizer, data_collator, verbalizer = verbalizer, template = templates)
    
    split = args.split 
    print(my_dataset[split].num_rows)
    args.num_examples = min(my_dataset[split].num_rows, args.num_examples )
    dataset = HuggingFaceDataset(my_dataset[split])
    
    attack_name = args.attack_name

    if attack_name == "none":
        model.mode = "eval"
        eval_results = evaluate_model(model, args)
        myprint (f"Results: {eval_results}")
    
    elif attack_name == "ood":
        ood_evaluation_loop(args, model)
    
    else:
        model.mode = "attack"
        attack_name_mapper = {"textfooler":TextFoolerCustom, 
                            "textbugger":TextBuggerCustom,
                            }
                            
        attack_class = attack_name_mapper[attack_name]
        log_to_csv=f"{args.model_dir}/{attack_name}_log.csv"

        print(templates)
        print(verbalizer)

        attack = attack_class.build(model)
        
        if args.query_budget < 0: args.query_budget = None 
        attack_args = AttackArgs(num_examples=args.num_examples, 
                                log_to_csv=log_to_csv, \
                                checkpoint_interval=100, 
                                checkpoint_dir=args.model_dir, 
                                disable_stdout=True,
                                query_budget = args.query_budget,parallel=False)
        attacker = Attacker(attack, dataset, attack_args)

        #set batch size of goal function
        attacker.attack.goal_function.batch_size = args.batch_size
        #set max words pertubed constraint
        max_percent_words = 0.1 if (args.dataset == "imdb" or args.dataset == "boolq" or args.dataset == "sst2" or args.dataset == "snli") else 0.3
        #flag = 0
        
        for i,constraint in enumerate(attacker.attack.constraints):
            if type(constraint) == textattack.constraints.overlap.max_words_perturbed.MaxWordsPerturbed:
                attacker.attack.constraints[i].max_percent = max_percent_words
            
        print(attacker)
       
        attacker.attack_dataset()
