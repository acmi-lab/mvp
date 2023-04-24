from random import triangular
from ..utils.funcs import *
import os, pickle, copy
import torch
from torch import nn
from .model_wrapper import ModelWrapper


class LPFT(ModelWrapper):
    def __init__(self, args, model, tokenizer, data_collator, dataset):
        super(LPFT, self).__init__(args, model, tokenizer, data_collator)
        if args.mode == "train" or args.model_type == "lp":
            model.cuda()
            
            train_features, train_labels = get_features_for_lpft(args, model, dataset["train"], model.device, args.cache_dir+"/"+args.model_type+"_"+args.dataset, split="train", batch_size = args.batch_size, tokenizer = tokenizer)
            val_features, val_labels = get_features_for_lpft(args, model, dataset["validation"], model.device, args.cache_dir+"/"+args.model_type+"_"+args.dataset, split="val", batch_size = args.batch_size, tokenizer = tokenizer)
            best_clf, best_coef, best_intercept, best_c, best_i, best_acc = linear_probe(train_features, None, val_features, train_labels, None, val_labels)
            with torch.no_grad():
                if args.num_labels > 2:
                    self.model.classifier.out_proj.weight.data = torch.Tensor(best_coef)
                    self.model.classifier.out_proj.bias.data = torch.Tensor(best_intercept)
                else:
                    self.model.classifier.out_proj.weight.data = torch.cat((torch.Tensor(best_coef), torch.Tensor(-1*best_coef)), 0)
                    self.model.classifier.out_proj.bias.data = torch.cat((torch.Tensor(best_intercept), torch.Tensor(-1*best_intercept)), 0)
        classifier_dropout = (
        model.config.classifier_dropout if model.config.classifier_dropout is not None else model.config.hidden_dropout_prob
            )
        self.dropout = nn.Dropout(classifier_dropout)
        
    def outs_to_logits(self, input_ids, outputs):
        if self.args.model_type == "lpft_sparse":
            hidden_states = outputs.hidden_states[-1]  #shape(hidden_states)  : B x seq_len x hidden_dim (768)
            cls_hidden_state = hidden_states[:,0,:]    #shape(cls_hidden_state)  : B x hidden_dim (768)
            cls_hidden_state = torch.tanh(cls_hidden_state)   #shape(cls_hidden_state)  : B x hidden_dim (768)
            cls_hidden_state = self.dropout(cls_hidden_state)  #shape(cls_hidden_state)  : B x hidden_dim (768)
            logits = self.model.classifier.out_proj(cls_hidden_state)           #shape(logits)  : B x num_labels
        else:
            logits = outputs.logits
        return logits