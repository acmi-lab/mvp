## Using HuggingFace modules to define basic skeletal

from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, AutoModelForMaskedLM, AutoModelForCausalLM, default_data_collator
from src.models import get_model
from src.utils import prepare_huggingface_dataset, tokenize_function, get_prompts, CustomTrainer, accuracy_metric, custom_eval
from tqdm import tqdm
import yaml
from transformers import EarlyStoppingCallback
import os
import math
import torch



def trainer(args):
    file = open(f"{args.model_dir}/tr_logs.txt", "a")  
    def myprint(a): print(a); file.write(a); file.write("\n"); file.flush()
    my_dataset, tokenizer, data_collator = prepare_huggingface_dataset(args)
    
    #Preprocess Function
    tokenized_dataset = my_dataset.map(tokenize_function(args, tokenizer), batched=True)

    verbalizer, templates = get_prompts(args)
    
    
    model = get_model(args, tokenized_dataset, tokenizer, data_collator, verbalizer = verbalizer, template = templates)
    interval = args.checkpoint_interval 
    if "mvp" in args.model_type:
        interval *= len(templates)
    interval = interval//torch.cuda.device_count()
    
    training_args = TrainingArguments(
        output_dir=args.model_dir,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.num_epochs,
        weight_decay=args.weight_decay,
        save_strategy="steps",
        report_to="none",
        evaluation_strategy = "steps",
        logging_steps = interval,
        save_steps = interval,
        metric_for_best_model = "eval_accuracy",
        greater_is_better = True,
        #https://github.com/huggingface/transformers/blob/504db92e7da010070c36e185332420a1d52c12b2/src/transformers/trainer.py#L626
        label_names = ['labels'],
        load_best_model_at_end=True,
        seed = args.seed,
        save_total_limit = 3,
    )
    #training_args.device = torch.device("cpu")
    print(templates)
    print(verbalizer)
    #using a custom trainer so that we can incorporate adversarial training. only difference is in keeping gradients on at eval time
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"], 
        tokenizer=tokenizer,
        data_collator=None,
        compute_metrics = accuracy_metric,
        callbacks = [EarlyStoppingCallback(early_stopping_patience=args.patience)],
        
    )
    # import ipdb
    # ipdb.set_trace()
    trainer.create_optimizer()
    #tokenized_dataset["test"] = tokenized_dataset["validation"] if (args.dataset=="sst2" or args.dataset=="boolq") else tokenized_dataset["test"]
    #custom training
    
    ## prepare loaders
    
    train_loader, val_loader, test_loader = trainer.get_train_dataloader(), trainer.get_eval_dataloader(), trainer.get_test_dataloader(tokenized_dataset["test"])
    train_results = trainer.train()
    accelerator = None
    
    myprint(f'Train Results: {train_results}')
    
    chkpoint_interval = int(math.ceil(my_dataset["train"].num_rows/args.batch_size/torch.cuda.device_count()))
    print(chkpoint_interval)
    
    save_directory = f"{args.model_dir}/final_model"
    tokenizer.save_pretrained(save_directory)
    model.model.save_pretrained(save_directory)
    
    for i in range(1, args.num_epochs+1):
        if i%5:
            os.system(f"rm -r {args.model_dir}/checkpoint-"+str(i*chkpoint_interval))
    model = torch.nn.DataParallel(model)
    eval_results = custom_eval(model, val_loader, accelerator)
    myprint(f'Val Results: {eval_results}')

    test_results = custom_eval(model, test_loader, accelerator)
    myprint(f'Test Results: {test_results}')
    exit(0)


