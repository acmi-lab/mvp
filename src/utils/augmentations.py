import numpy as np
import torch
import torch.nn as nn


def criterion_CE(combined_model, perturbed, attention_mask, original, input_ids, labels):
    adv_inputs = {}
    adv_inputs['inputs_embeds'] = perturbed
    adv_inputs['attention_mask'] = attention_mask
    adv_inputs['output_attentions'] = True
    outputs = combined_model.model(**adv_inputs)
    logits = combined_model.outs_to_logits(input_ids, outputs)
    return nn.CrossEntropyLoss()(logits, labels)


def pgd_attack(combined_model, input_ids, attention_mask, y, params, norm="linf", criterion = criterion_CE):
    # "norm": may be linf or l2
    model = combined_model.model
    is_training = model.training
    if is_training:
        model.eval()    # Need to freeze the batch norm and dropouts unles specified not to
    
    word_embedding_layer = model.get_input_embeddings()
    X = word_embedding_layer(input_ids).detach()
    assert(X.requires_grad == False)

    epsilon = params.epsilon
    alpha = params.alpha
    num_iter = params.num_iter

    if alpha == None:
        alpha = (epsilon / num_iter)*1.5
   
    
    if norm == "linf":
        delta = torch.empty_like(X).uniform_(-epsilon, epsilon)
    else:
        assert(norm == "l2")
        delta = torch.rand_like(X, requires_grad=True) 
        delta.data *= (2.0*delta.data - 1.0)
        delta.data = delta.data*epsilon/norms(delta.detach()) 

    delta.requires_grad = True

    for _ in range(num_iter):
        loss = criterion(combined_model, X+delta, attention_mask, X, input_ids, y)
        loss.backward()

        if norm == "linf":
            delta.data = (delta.data + alpha*delta.grad.detach().sign()).clamp(-epsilon,epsilon)
        else:
            assert(norm == "l2")
            delta.data +=  alpha*delta.grad.detach() / norms(delta.grad.detach())
            delta.data *= epsilon / norms(delta.detach()).clamp(min=epsilon)
            
        delta.grad.zero_()

    if is_training:
        model.train()    #Reset to train mode if model was training earlier
    return delta + word_embedding_layer(input_ids)

def norms(Z):
    return Z.view(Z.shape[0], -1).norm(dim=1)[:,None,None]
