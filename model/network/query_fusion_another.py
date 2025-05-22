import torch
import torch.nn as nn
from transformers import BertModel, BertConfig
import torch.nn.functional as F

class MultimodalBertWithMaskedQuery(nn.Module):
    def __init__(self, config, num_query_tokens=4):
        super().__init__()
        self.config = config
        self.bert = BertModel(config, add_pooling_layer=False)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.query_tokens = nn.Parameter(torch.randn(num_query_tokens, config.hidden_size))
        self.num_query_tokens = num_query_tokens
        self.max_position_embeddings = config.max_position_embeddings

    def forward(self, audio_input, text_input, audio_mask, text_mask):
        B = audio_input.size(0)
        device = audio_input.device

        # [CLS] and [SEP]
        cls_token = torch.zeros(B, 1, audio_input.size(-1)).to(device)
        sep_token = torch.zeros(B, 1, audio_input.size(-1)).to(device)

        audio_input = torch.cat([cls_token, audio_input, sep_token], dim=1)  # [B, L1+2, D]
        text_input = torch.cat([text_input, sep_token], dim=1)  # [B, L2+1, D]
        input_embeds = torch.cat([audio_input, text_input], dim=1)  # [B, L, D]

        L = input_embeds.size(1)
        query_tokens = self.query_tokens.unsqueeze(0).expand(B, -1, -1)  # [B, Q, D]
        full_input = torch.cat([input_embeds, query_tokens], dim=1)  # [B, L+Q, D]

        # ===== 构造 Attention Mask（B, 1, L+Q, L+Q） =====
        total_len = full_input.size(1)
        full_mask = torch.ones(B, total_len, total_len).to(device)  # 默认全允许

        # 禁止 query-token 之间 attention
        query_start = L
        for i in range(self.num_query_tokens):
            for j in range(self.num_query_tokens):
                if i != j:
                    full_mask[:, query_start + i, query_start + j] = 0

        # 添加模态 attention mask
        audio_len = audio_input.size(1)
        text_len = text_input.size(1)
        audio_mask = audio_mask.to(device)
        text_mask = text_mask.to(device)
        modal_mask = torch.cat([
            torch.ones(B, 1).to(device),   # CLS
            audio_mask,                   # audio
            torch.ones(B, 1).to(device),  # SEP
            text_mask,                    # text
            torch.ones(B, 1).to(device),  # SEP
            torch.ones(B, self.num_query_tokens).to(device)  # query tokens
        ], dim=1)  # [B, L+Q]

        attention_mask = modal_mask.unsqueeze(1) * full_mask  # (B, 1, L+Q, L+Q)

        outputs = self.bert(
            inputs_embeds=full_input,
            attention_mask=attention_mask,
            return_dict=True
        )

        cls_output = outputs.last_hidden_state[:, 0, :]  # [CLS]
        logits = self.classifier(cls_output)
        return logits
