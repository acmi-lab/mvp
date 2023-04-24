from src.models.model_wrapper import ModelWrapper
import torch

class CLSPrompt(ModelWrapper):
    def __init__(self, args, model, tokenizer, data_collator, verbalizer = None, template=None,):
        label_words = [item for sublist in verbalizer.values() for item in sublist]
        label_set = []
        self.verbalizer = verbalizer
        self.tokenizer = tokenizer
        
        #"gpt" does not have cls token
        assert ('gpt' not in args.model)
        num_tokens = 3 # bert tokenizes into cls, word, sep we want the word to be a single token
        for k,v in self.verbalizer.items():
            for word in v:
                if "roberta" in args.model:
                    word = " " + word
                if(len(self.tokenizer(word)["input_ids"]) == num_tokens):
                    label_set.append(k)
                else:
                    print(word)
        self.label_set = torch.tensor(label_set)
        toks = self.tokenizer(label_words)["input_ids"]
        new_toks = [t for t in toks if len(t) == num_tokens]
        self.label_word_ids = torch.tensor(new_toks)[:,1]
        super(CLSPrompt, self).__init__(args, model, tokenizer, data_collator)

    def outs_to_logits(self, input_ids, outputs):
        logits = outputs.logits
        batchid, indices = torch.where(input_ids == self.tokenizer.cls_token_id)

        cls_mask_logits = logits[batchid, indices,:]
        label_words_logits = cls_mask_logits[:, self.label_word_ids]

        return label_words_logits