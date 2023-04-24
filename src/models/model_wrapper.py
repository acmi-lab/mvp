from random import triangular
from ..utils import *
import os, pickle, copy
import torch
from torch import nn

class ModelWrapper(torch.nn.Module):
    def __init__(self, args, model, tokenizer, data_collator, verbalizer = None, template=None):  
        '''
        args: args object from argparse
        model: huggingface model
        tokenizer: huggingface tokenizer
        data_collator: huggingface data collator
        verbalizer: verbalizer object
        template: list of templates

        This is a wrapper class for the huggingface model. It is used to add the verbalizer and template to the model.
        '''

        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.mode = args.mode
        self.data_collator = data_collator
        self.args = args
        self.verbalizer = verbalizer
        self.template = template
        if 'gpt' in self.args.model:
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
            self.model.config.mask_token_id = self.tokenizer.mask_token_id
        self.config = self.model.config
        self.model.resize_token_embeddings(len(tokenizer))
    
    
    def text_to_ids(self, text):
        '''
        text: list of strings

        returns: input_ids, attention_mask
        '''
        inputs_dict = self.tokenizer(text,
            add_special_tokens=True,
            padding=False,
            truncation=True,
        )
        inputs_dict = self.data_collator(inputs_dict)
        input_ids = inputs_dict["input_ids"].cuda()
        attention_mask = inputs_dict["attention_mask"].cuda()
        return input_ids, attention_mask

    
    def get_updated_input_ids(self, input_ids, attention_mask, **kwargs):
        '''
        input_ids: torch tensor of shape (batch_size, seq_len)
        attention_mask: torch tensor of shape (batch_size, seq_len)
        kwargs: additional arguments

        returns: updated input_ids, attention_mask after adding the verbalizer and template in case of mvp
        '''
        ## if we receive tokenized text, untokenize it:
        if type(input_ids) != type(torch.ones(1)):
            text_input_list = []
            for t in input_ids:
                if type(t) is not tuple:
                    text_input_list.append(t.lower())
                elif self.args.dataset == "boolq":
                    text_input_list.append((t[1]+"</s></s>"+t[0]+"?").lower()) 
                else:
                    text_input_list.append((t[0]+"</s></s>"+t[1]).lower())
            input_ids, attention_mask = self.text_to_ids(text_input_list)


        if self.args.model_type == "mvp" or self.args.model_type == "untrained_mvp":
            input_ids, attention_mask = insert_tokenized_prompts(self.tokenizer, self.args.model, input_ids, self.template, self.len_templates, use_all = (self.args.num_template != -2) or self.mode!="train")
        return input_ids, attention_mask
    

    def forward(self, input_ids, attention_mask=None, **kwargs):
        '''
        input_ids: torch tensor of shape (batch_size, seq_len)
        attention_mask: torch tensor of shape (batch_size, seq_len)
        kwargs: additional arguments

        returns: logits
        '''
        if 'label' in kwargs.keys(): 
            #for gpt2 model
            kwargs['labels'] = kwargs['label']
        input_ids, attention_mask  = self.get_updated_input_ids(input_ids, attention_mask, **kwargs)
        input_ids, attention_mask = input_ids.to(self.model.device), attention_mask.to(self.model.device)
        
        if self.args.adv_augment and self.mode=="train":
            embedding_outs = pgd_attack(self, input_ids, attention_mask, kwargs['labels'], self.args, norm = self.args.norm)
            adv_inputs = {}
            adv_inputs['inputs_embeds'] = embedding_outs
            adv_inputs['attention_mask'] = attention_mask
            adv_inputs['output_attentions'] = True
            outputs = self.model(**adv_inputs)
        else:
            outputs = self.model(input_ids=input_ids, attention_mask = attention_mask, output_hidden_states = True, output_attentions = True)
        
        logits = self.outs_to_logits(input_ids, outputs)

        if 'labels' in kwargs.keys() and self.mode in ["train", "eval"]:
            loss = F.cross_entropy(logits, kwargs['labels'])
            return loss, logits
        else:
            assert (self.mode == "attack")
            return logits.detach()