import torch
import torch.nn as nn
import torch.nn.functional as F

class ModalityCentricAugmentation(nn.Module):
    def __init__(self, d_model, n_heads, d_hid):
        super(ModalityCentricAugmentation, self).__init__()
        self.intra_attn = nn.MultiheadAttention(d_model, n_heads)
        self.inter_attn = nn.MultiheadAttention(d_model, n_heads)
        self.msmha = MultiScaleMultiHeadAttention(d_model, n_heads, d_hid)
        self.ffn = nn.Linear(d_model, d_model)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, center_modality, aux_modality):
        # Intra-Modality Attention
        intra_attn_output, _ = self.intra_attn(center_modality, center_modality, center_modality)
        
        # Inter-Modality Attention
        inter_attn_output, _ = self.inter_attn(center_modality, aux_modality, aux_modality)
        inter_attn_output = self.msmha(inter_attn_output)
        
        # Combine Intra- and Inter-Modality Attention
        combined_output = intra_attn_output + inter_attn_output
        combined_output = self.layer_norm(combined_output)
        combined_output = self.ffn(combined_output)
        
        return combined_output

class MultiScaleMultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_hid):
        super(MultiScaleMultiHeadAttention, self).__init__()
        self.local_conv1 = nn.Conv1d(d_model, d_hid, kernel_size=1)
        self.local_conv2 = nn.Conv1d(d_hid, d_model, kernel_size=1)
        self.global_conv1 = nn.Conv1d(d_model, d_hid, kernel_size=1)
        self.global_conv2 = nn.Conv1d(d_hid, d_model, kernel_size=1)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.batch_norm = nn.BatchNorm1d(d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        local_output = self.local_conv2(self.relu(self.local_conv1(x)))
        global_output = self.global_conv2(self.relu(self.global_conv1(self.global_avg_pool(x))))
        return x * torch.sigmoid(local_output + global_output)