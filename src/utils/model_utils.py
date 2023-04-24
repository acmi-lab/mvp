
import torch
import random

def insert_tokenized_template_front(tokenizer, model_type, input_id, template_id, len_templates):
    '''
    input_id: (batch, token_length)
    template_id: (token_length,)
    '''

    if "roberta" in model_type:
        template_id = tokenizer(" ")["input_ids"][1:-1] + template_id
    new_input_id = torch.zeros(min(tokenizer.model_max_length, input_id.shape[0]+max(len_templates)))
    new_input_id[0] = tokenizer.cls_token_id
    pad_indices = 1*(input_id==tokenizer.pad_token_id).nonzero()
    first_pad_index = 0
    if pad_indices.shape[0] > 0:
        first_pad_index = pad_indices[0].item()
    else:
        first_pad_index = input_id.shape[0]
    if(first_pad_index + len(template_id) < tokenizer.model_max_length):
        new_input_id[1:len(template_id)+1] = torch.tensor(template_id)
        new_input_id[len(template_id)+1:first_pad_index+len(template_id)] = input_id[1:first_pad_index]
        if first_pad_index+len(template_id) < new_input_id.shape[0]:
            new_input_id[first_pad_index+len(template_id):] = torch.tensor([tokenizer.pad_token_id]*new_input_id[first_pad_index+len(template_id):].shape[0])
    else:
        new_input_id[1:len(template_id)+1] = torch.tensor(template_id)
        new_input_id[len(template_id)+1:] = input_id[1:tokenizer.model_max_length - len(template_id)]
        new_input_id[-1] = tokenizer.sep_token_id
    new_attention_mask = 1*(new_input_id !=  tokenizer.pad_token_id)

    return new_input_id, new_attention_mask

def insert_tokenized_template_back(tokenizer, model_type, input_id, template_id, len_templates):
    '''
    input_id: (batch, token_length)
    template_id: (token_length,)
    '''
    if "roberta" in model_type:
        template_id = tokenizer(" ")["input_ids"][1:-1] + template_id
    # create a new input d initialized with pad token id
    # new_input -> (token_length,)
    new_input_id = torch.ones(min(tokenizer.model_max_length, input_id.shape[0]+max(len_templates)))*tokenizer.pad_token_id
    # add cls token at the start
    if "gpt" not in model_type:
        new_input_id[0] = tokenizer.cls_token_id 
    # find out all the pad_indices in the input_id
    pad_indices = 1*(input_id==tokenizer.pad_token_id).nonzero()
    first_pad_index = 0
    # find out the first pad index. If no pad then use the last sep token
    if pad_indices.shape[0] > 0:
        first_pad_index = pad_indices[0].item() - 1
    else:
        first_pad_index = input_id.shape[0] - 1
    if(first_pad_index + len(template_id) + 1 < tokenizer.model_max_length):
        new_input_id[:first_pad_index] = input_id[:first_pad_index]
        new_input_id[first_pad_index:first_pad_index+len(template_id)] = torch.tensor(template_id)
        if "gpt" not in model_type:
            new_input_id[first_pad_index+len(template_id)] = tokenizer.sep_token_id
        if first_pad_index+len(template_id) < new_input_id.shape[0]:
            new_input_id[first_pad_index+len(template_id)+1:] = torch.tensor([tokenizer.pad_token_id]*new_input_id[first_pad_index+len(template_id)+1:].shape[0])
    else:
        new_input_id[:tokenizer.model_max_length-len(template_id)-1] = input_id[:tokenizer.model_max_length-len(template_id)-1]
        new_input_id[tokenizer.model_max_length-len(template_id)-1:tokenizer.model_max_length-1] = torch.tensor(template_id)
        if "gpt" not in model_type:
            new_input_id[-1] = tokenizer.sep_token_id

    new_attention_mask = 1*(new_input_id !=  tokenizer.pad_token_id)
    return new_input_id, new_attention_mask


def insert_tokenized_prompts(tokenizer, model_type, text_input_list, templates, len_templates, use_all=True):
    #input_ids, attention_mask = self.text_to_ids(text_input_list)]
    input_ids = text_input_list
    num_templates_used = len(templates) if use_all else 1
    new_input_ids = torch.zeros(input_ids.shape[0]*num_templates_used, min(tokenizer.model_max_length, input_ids.shape[1]+max(len_templates)))
    new_attention_masks = torch.zeros(input_ids.shape[0]*num_templates_used, min(tokenizer.model_max_length, input_ids.shape[1]+max(len_templates)))
    for i in range(input_ids.shape[0]):
        if use_all:
            templates_new = templates
        else:
            templates_new = random.choices(templates, k=1)
        j = 0
        for template in templates_new:
            template = template.replace("[MASK]", tokenizer.mask_token)
            if template.split(" ")[0] == "[SEP]":
                template_ids = tokenizer(" ".join(template.split(" ")[1:]))["input_ids"][1:-1] if "gpt" not in model_type else tokenizer(" ".join(template.split(" ")[1:]))["input_ids"]
                new_input_id, new_attention_mask = insert_tokenized_template_back(tokenizer, model_type, input_ids[i,:], template_ids, len_templates)
            else:
                template_ids = tokenizer(template)["input_ids"][1:-1] if "gpt" not in model_type else tokenizer(template)["input_ids"]
                new_input_id, new_attention_mask = insert_tokenized_template_front(tokenizer, model_type, input_ids[i,:], template_ids, len_templates)
            
            new_input_ids[num_templates_used*i+j,:] =  new_input_id
            new_attention_masks[num_templates_used*i+j,:] =  new_attention_mask
            j=j+1
    return new_input_ids.long(), new_attention_masks.long()



