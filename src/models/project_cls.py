from .model_wrapper import ModelWrapper
from torch import nn
import torch

class ProjectCLS(ModelWrapper):
    def __init__(self, args, model, tokenizer, data_collator):
        super(ProjectCLS, self).__init__(args, model, tokenizer, data_collator)
        classifier_dropout = (
            model.config.classifier_dropout if model.config.classifier_dropout is not None else model.config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
    def outs_to_logits(self, input_ids, outputs):
        hidden_states = outputs.hidden_states[-1]  #shape(hidden_states)  : B x seq_len x hidden_dim (768)
        cls_hidden_state = hidden_states[:,0,:]    #shape(cls_hidden_state)  : B x hidden_dim (768)
        cls_hidden_state = torch.tanh(cls_hidden_state)   #shape(cls_hidden_state)  : B x hidden_dim (768)
        cls_hidden_state = self.dropout(cls_hidden_state)  #shape(cls_hidden_state)  : B x hidden_dim (768)
        logits = self.model.classifier.out_proj(cls_hidden_state)           #shape(logits)  : B x num_labels
        return logits