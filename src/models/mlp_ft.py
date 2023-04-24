from .model_wrapper import ModelWrapper

class MLP_FT(ModelWrapper):
    def __init__(self, args, model, tokenizer, data_collator):
        super(MLP_FT, self).__init__(args, model, tokenizer, data_collator)

    def outs_to_logits(self, input_ids, outputs):
        return outputs.logits