import torch
import numpy as np
import random
import torch.nn.functional as F
from nltk.corpus import wordnet
import textattack
import copy
import yaml
from torch.utils.data import random_split
from torch.utils.data import Dataset
import re
from textattack.shared import WordEmbedding
from transformers import DataCollatorWithPadding, DataCollatorForLanguageModeling, default_data_collator
import pandas as pd
from sklearn.linear_model import LogisticRegression
import os
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, TrainingArguments

counterfitted_glove_embedding = WordEmbedding.counterfitted_GLOVE_embedding()
   
       


def tokenize_function(args, tokenizer):
    text_datasets = {"imdb":"text",
                    "amazon_polarity":"content",
                    "ag_news":"text",
                    "rotten_tomatoes":"text",
                    "emotion":"text",
                    "emotion2":"text",
                    "movie_rationales":"review",
                    "sst2":"sentence"}
    
    def ret_func(examples):
        if args.dataset in text_datasets.keys():
            #classification
            return tokenizer(examples[text_datasets[args.dataset]], truncation=True)
        if  args.dataset in ["mrpc"]:
            return tokenizer(examples["sentence1"], examples["sentence2"], truncation=True)
        if  args.dataset in ["snli"]:
            return tokenizer(examples["premise"], examples["hypothesis"], truncation=True)
        if  args.dataset in ["boolq"]:
            return tokenizer(examples["sentence"], examples["question"], truncation=True)
        raise("Unsupported")
    return ret_func


def do_lower(example):
    for key in example.keys():
        if type(example[key]) == type("str"):
        # if key not in ["label", "answer", "evidences"]:
            example[key] = example[key].lower()
    return example

def prepare_huggingface_dataset(args):
    from datasets import load_dataset
    #ipdb.set_trace()
    name = "emotion" if args.dataset == "emotion2" else args.dataset
    my_dataset = load_dataset(name) if name!="sst2" else load_dataset("glue",name)
    my_dataset = my_dataset.map(do_lower)
    
    if args.dataset == "sst2":
        # test datset has no labels, so use val set as test for sst2
        my_dataset["test"] = my_dataset["validation"]
        assert args.val_size + args.train_size <= 1
        train_dataset, validation_dataset= my_dataset["train"].train_test_split(test_size=args.val_size, train_size = args.train_size, seed = 0).values()
        my_dataset["train"] = train_dataset
        my_dataset["validation"] = validation_dataset
    elif "test" not in my_dataset.keys() and "validation" in my_dataset.keys() :
        test, val = my_dataset["validation"].train_test_split(test_size=0.5, train_size = 0.5, seed = 0).values()
        my_dataset["validation"] = val
        my_dataset["test"] = test
        if args.train_size < 1:
            train_dataset, _= my_dataset["train"].train_test_split(test_size=args.val_size, train_size = args.train_size, seed = 0).values()
            my_dataset["train"] = train_dataset
    elif "validation" not in my_dataset.keys():
        assert args.val_size + args.train_size <= 1
        train_dataset, validation_dataset= my_dataset["train"].train_test_split(test_size=args.val_size, train_size = args.train_size, seed = 0).values()
        my_dataset["train"] = train_dataset
        my_dataset["validation"] = validation_dataset
    else:
        if args.train_size < 1:
            train_dataset, _= my_dataset["train"].train_test_split(test_size=args.val_size, train_size = args.train_size, seed = 0).values()
            my_dataset["train"] = train_dataset
    
    if args.dataset == "sst2":
        my_dataset = my_dataset.filter(lambda example: (re.search('[a-zA-Z]', example["sentence"]) is not None))
    if args.dataset == "snli":
        # Dataset instances which don't have any gold label are marked with -1 label. 
        # Make sure you filter them before starting the training using datasets.Dataset.filter.
        # This is too slow!
        my_dataset = my_dataset.filter(lambda example: (example['label']!=-1))
    if args.dataset == "boolq":
        def map_labels(example):
            key_map_dict = {'passage':'sentence','answer':'label'}
            example = {(key_map_dict[k] if k in key_map_dict else k):v  for (k,v) in example.items() }
            new_mapping = {False:0, True:1}
            example['label'] = new_mapping[example['label']]
            return example
        my_dataset = my_dataset.map(map_labels)
    if args.dataset == "emotion2":
        # old labels sadness (0), joy (1), love (2), anger (3), fear (4), surprise (5).
        # new labels sadness + anger (0) | joy + love (1). Drop rest
        #filter out 4 and 5
        my_dataset = my_dataset.filter(lambda example: example['label'] <4)

        #map 2 to 1 and 3 to 0
        def map_labels(example):
            new_mapping = {0:0, 1:1, 2:1, 3:0}
            example['label'] = new_mapping[example['label']]
            return example
        my_dataset = my_dataset.map(map_labels)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, do_lower_case = True, model_max_length = args.max_length)
    tokenizer.model_max_length = args.max_length
    #does not work. some bug https://github.com/huggingface/transformers/issues/17675
    tokenizer.do_lower_case = True
    # #Padding
    if 'gpt' in args.model:
        tokenizer.add_special_tokens({'pad_token': '<|pad]|>', 'mask_token':'<|mask|>'})
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        
    return my_dataset, tokenizer, data_collator

from datasets import load_metric
metric = load_metric('accuracy')

def get_features_for_lpft(args, text_encoder, dataset, device, cache_dir, noscale=True, split="train", batch_size = 8, tokenizer = None):
    text_encoder.train()
    if os.path.exists(os.path.join(cache_dir, split+"_features.pt")):
        return torch.load(os.path.join(cache_dir, split+"_features.pt")), torch.load(os.path.join(cache_dir, split+"_labels.pt"))
    else:
        os.makedirs(cache_dir, exist_ok=True)
    from src.utils import CustomTrainer
    training_args = TrainingArguments(
        output_dir="./checkpoints/cache_dir",
        per_device_train_batch_size=batch_size,
    )
    trainer = CustomTrainer(
        model=text_encoder,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        data_collator=None,
        
    )
    dataloader = trainer.get_train_dataloader()
    batch_id = 0
    for batch in tqdm(dataloader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch['attention_mask'].to(device)
        targets = batch['labels'].to(device)
        outputs = text_encoder(input_ids, attention_mask, labels = targets, output_hidden_states = True)
        hidden_states = outputs.hidden_states[-1]
        hidden_states = hidden_states[:,0,:]
        hidden_states = torch.tanh(hidden_states)
        if args.model_type == "lpft_dense":
                hidden_states = text_encoder.classifier.dense(hidden_states)
        hidden_states = hidden_states.detach().cpu()
        if batch_id == 0:
            features = hidden_states
            features = features.detach().cpu()
            labels = targets
            labels = labels.detach().cpu()
        else:
            features = torch.cat((features, hidden_states), 0)
            features = features.detach().cpu()
            targets = targets.detach().cpu()
            labels = torch.cat((labels, targets))
            labels = labels.detach().cpu()
        batch_id+=1
    torch.save(features, os.path.join(cache_dir, split+"_features.pt"))
    torch.save(labels, os.path.join(cache_dir, split+"_labels.pt"))
    return features, labels

def accuracy_metric(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    outs = metric.compute(predictions=predictions, references=labels)
    return outs

def evaluate_model(model, args):
    model.eval()
    my_dataset, tokenizer,data_collator = prepare_huggingface_dataset(args)
    tokenized_dataset = my_dataset.map(tokenize_function(args, tokenizer), batched=True)
    from transformers import Trainer, TrainingArguments

    
    training_args = TrainingArguments(output_dir = args.model_dir,
    per_device_eval_batch_size=args.batch_size*4, label_names = ["labels"])
    
    trainer = Trainer(
        model=model,
        eval_dataset=tokenized_dataset["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics = accuracy_metric,
        args = training_args
    )

    with torch.no_grad():
        eval_results = trainer.evaluate()
    return eval_results

def get_prompts(args):
    verbalizer = yaml.safe_load(open(args.verbalizer_file,'r'))
    templates = list(yaml.safe_load(open(args.template_file,'r')).values())
    if args.num_template >= 0:
        templates = [templates[args.num_template]]
    return verbalizer, templates

from tqdm import tqdm

def custom_train(model, loader, optimizer, scheduler, accelerator, num_epochs = 5, patience = 5, num_evals_per_epoch = 5, eval_loader = None):
    #evaluate the model num_evals_per_epoch times through the course of each epoch
    eval_every_steps = len(loader)//num_evals_per_epoch
    
    #patience: if the eval accuracy does not decrease for 5 consecutive evaluations, stop training
    patience_counter = 0
    prev_eval_acc = 0
    end_training = 0

    model.train()
    device = accelerator.device if accelerator is not None else 'cuda'
    model.to(device)
    
    
    for epoch in range(1,num_epochs+1):
        if end_training: break
        
        cumulative_accuracy = 0
        n_examples = 0
        with tqdm(loader, unit="batch") as tepoch:
            tepoch.set_description(f"Epoch {epoch}")
            for num_steps,batch in enumerate(tepoch):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch['attention_mask'].to(device)
                targets = batch['labels'].to(device)
                loss, logits = model(input_ids, attention_mask, labels = targets)
                
                preds = logits.argmax(1).detach()
                accuracy = (preds == targets).sum()
                
                cumulative_accuracy += accuracy 
                n_examples += preds.shape[0]

                accelerator.backward(loss)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                model.zero_grad()
                tepoch.set_postfix(loss=loss.item(), accuracy= accuracy.item()/ preds.shape[0])
                

                ## EVALUATION CONDITION
                if (num_steps >= 0) and (num_steps%eval_every_steps == 0) and (eval_loader is not None):
                    print(model.model.roberta.encoder.layer[-1].attention.self.query.weight)
                    current_eval_accuracy = custom_eval(model, eval_loader, accelerator)
                    if current_eval_accuracy <= prev_eval_acc:
                        patience_counter += 1
                        if patience_counter == patience: end_training = 1; break
                    else:
                        patience_counter = 0
                    prev_eval_acc = current_eval_accuracy
    
    return cumulative_accuracy/n_examples


def custom_eval(model, loader, accelerator):
    model.eval()
    device = accelerator.device if accelerator is not None else 'cuda'
    model.to(device)
    cumulative_accuracy = 0
    n_examples = 0
    
    # with torch.no_grad():
    with tqdm(loader, unit="batch") as tepoch:
        tepoch.set_description(f"Evaluating ")
        for batch in tepoch:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch['attention_mask'].to(device)
            targets = batch['labels'].to(device)
            loss, logits = model(input_ids, attention_mask, labels = targets)
            loss = loss.mean()
            preds = logits.argmax(1).detach()
            accuracy = (preds == targets).sum().detach().cpu()
            
            cumulative_accuracy += accuracy 
            n_examples += preds.shape[0]
            print(loss, cumulative_accuracy)
            tepoch.set_postfix(loss=loss.item(), accuracy= cumulative_accuracy.item()/n_examples)
    

    return cumulative_accuracy/n_examples

def linear_probe(train_features,
                               val_features,
                               test_features,
                               train_labels,
                               val_labels,
                               test_labels,
                               num_cs=50,
                               start_c=-1,
                               end_c=2,
                               max_iter=500,
                               random_state=0):
    Cs = np.logspace(start_c, end_c, num_cs)
    clf = LogisticRegression(random_state=random_state,
                             warm_start=True,
                             max_iter=max_iter, multi_class = 'multinomial')
    best_acc = -1.0
    best_clf, best_coef, best_intercept, best_i, best_c = None, None, None, None, None
    for i, C in zip(range(len(Cs)), Cs):
        clf.C = C
        clf.fit(train_features, train_labels)
        # ipdb.set_trace()
        corrects = np.array(clf.predict(test_features)) ==  np.array(test_labels)
        test_acc = corrects.mean()
        print(f"i : {i} c: {C} Val Acc : {0} Test Acc : {test_acc}")
        # logger.info(f"i : {i} c: {C} Val Acc : {0} Test Acc : {test_acc}")
        if test_acc > best_acc:
            best_acc = test_acc
            best_clf = copy.deepcopy(clf)
            best_coef = copy.deepcopy(clf.coef_)
            best_intercept = copy.deepcopy(clf.intercept_)
            best_i = i
            best_c = C
    print(f"best c: {best_c} best Val Acc : {0} best Test Acc : {best_acc}")
    return best_clf, best_coef, best_intercept, best_c, best_i, best_acc