import torch
import torch.nn as nn
import torchaudio
from transformers import RobertaTokenizer, RobertaModel

class TextEncoder(nn.Module):
    def __init__(self, tokenizer=None, text_model=None, model_path=None):
        super(TextEncoder, self).__init__()
        if model_path is None:
            self.tokenizer = tokenizer
            self.text_model = text_model
        else:
            self.tokenizer = RobertaTokenizer.from_pretrained(model_path)
            self.text_model = RobertaModel.from_pretrained(model_path)
            
    def forward(self, text_inputs):
        device=next(self.parameters()).device
        text_embeddings = self.tokenizer(text_inputs, return_tensors='pt', padding=True, truncation=True,
                                         max_length=80)
        text_output = self.text_model(input_ids=text_embeddings["input_ids"].to(device),attention_mask=text_embeddings["attention_mask"].to(device))
        # text_output = self.text_model(**text_embeddings)
        text_seq_embedding, text_cls_embeddings = text_output[0], text_output[1]
        # print('text_encoder',text_seq_embedding.shape, text_cls_embeddings.shape,text_embeddings["attention_mask"].shape,text_embeddings["attention_mask"])
        return text_seq_embedding, text_cls_embeddings#, text_embeddings["attention_mask"]
    