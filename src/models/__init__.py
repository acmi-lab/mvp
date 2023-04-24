from .clsprompt import CLSPrompt
from .lpft import LPFT
from .mlp_ft import MLP_FT
from .mvp import MVP
from .project_cls import ProjectCLS
from transformers import AutoModelForSequenceClassification, AutoModelForMaskedLM, AutoModelForCausalLM, T5ForConditionalGeneration
import os
import torch

def get_model(args, dataset, tokenizer, data_collator, verbalizer = None, template = None):
    if args.path != "None": 
        location = args.path
    elif args.path == "None":
        location = args.model
    else:
        raise Exception("path not available")
    if args.mode == "train":
        regime = "train"
    else:
        regime = "test"
    #model_class = AutoModelForCausalLM if "gpt" in args.model elif "t5" in args.model else AutoModelForMaskedLM
    if "gpt" in args.model:
        model_class = AutoModelForCausalLM
    else:
        model_class = AutoModelForMaskedLM

    if os.path.exists(f"{location}pytorch_model.bin"):
        weights = torch.load(f"{location}pytorch_model.bin")
        if "model." in list(weights.keys())[0]:
            new_weights = copy.deepcopy(weights)
            for key in weights.keys():
                new_key = key[6:]
                new_weights[new_key] = weights[key]
                del new_weights[key]
            torch.save(new_weights, f"{location}pytorch_model.bin")
        
    if args.model_type == 'mvp' or (args.model_type == "untrained_mvp" and regime=="test") or (args.model_type=="untrained_mvp" and args.path!="None"):
        base_model = model_class.from_pretrained(location, return_dict = True)
        model = MVP(args, base_model, tokenizer, data_collator, verbalizer = verbalizer, template = template)
    elif args.model_type == 'clsprompt' or (args.model_type == "untrained_clsprompt" and regime=="test") or (args.model_type=="untrained_clsprompt" and args.path!="None"):
        base_model = model_class.from_pretrained(location, return_dict = True)
        model = CLSPrompt(args, base_model, tokenizer, data_collator, verbalizer = verbalizer, template = template)
    elif args.model_type == "untrained_mvp" and regime == "train" and args.path == "None":
        from transformers import AutoConfig
        config = AutoConfig.for_model("roberta")
        config.layer_norm_eps = 1e-5
        config.max_position_embeddings = 514
        config.type_vocab_size = 1
        config.vocab_size = 50265 
        base_model = AutoModelForMaskedLM.from_config(config)
        model = MVP(args, base_model, tokenizer, data_collator, verbalizer = verbalizer, template = template)
    elif args.model_type == "untrained_clsprompt" and regime == "train" and args.path == "None":
        from transformers import AutoConfig
        config = AutoConfig.for_model("roberta")
        config.layer_norm_eps = 1e-5
        config.max_position_embeddings = 514
        config.type_vocab_size = 1
        config.vocab_size = 50265 
        base_model = AutoModelForMaskedLM.from_config(config)
        model =CLSPrompt(args, base_model, tokenizer, data_collator, verbalizer = verbalizer, template = template)
    elif args.model_type == "untrained_mlp_ft" and regime == "train" and args.path == "None":
        from transformers import AutoConfig
        config = AutoConfig.for_model("roberta")
        config.layer_norm_eps = 1e-5
        config.max_position_embeddings = 514
        config.type_vocab_size = 1
        config.vocab_size = 50265 
        config.num_labels = args.num_labels
        base_model = AutoModelForSequenceClassification.from_config(config=config)
        model = MLP_FT(args, base_model, tokenizer, data_collator)
    elif args.model_type == "projectcls":
        base_model = AutoModelForSequenceClassification.from_pretrained(location, num_labels=args.num_labels)
        model = ProjectCLS(args, base_model, tokenizer, data_collator)
    elif "lpft" in args.model_type:
        base_model = AutoModelForSequenceClassification.from_pretrained(location, num_labels=args.num_labels)
        model = LPFT(args, base_model, tokenizer, data_collator, dataset)
    else:
        base_model = AutoModelForSequenceClassification.from_pretrained(location, num_labels=args.num_labels)
        model = MLP_FT(args, base_model, tokenizer, data_collator)
    return model