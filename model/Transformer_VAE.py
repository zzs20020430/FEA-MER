import torch
import torch.nn as nn
import torch.nn.functional as F


def get_diffusion_params(T, beta_start=1e-4, beta_end=0.02):
    betas = torch.linspace(beta_start, beta_end, T)
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    return betas, alphas, alphas_cumprod

class TransformerEncoder(nn.Module):
    def __init__(self, hidden_dim, latent_dim, num_layers=6, nhead=8):
        super(TransformerEncoder, self).__init__()
        self.input_proj = nn.Linear(hidden_dim, latent_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=latent_dim, nhead=nhead, dim_feedforward=latent_dim*4)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        # x: (B, D) --> (B, 1, D) --> (1, B, D) for transformer
        x = self.input_proj(x).unsqueeze(1).transpose(0, 1)
        x = self.transformer(x)
        return x.squeeze(0)

class TransformerDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, num_layers=6, nhead=8):
        super(TransformerDecoder, self).__init__()
        decoder_layer = nn.TransformerEncoderLayer(d_model=latent_dim, nhead=nhead, dim_feedforward=latent_dim*4)
        self.transformer = nn.TransformerEncoder(decoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(latent_dim, hidden_dim)

    def forward(self, x):
        # x: (B, D) --> (1, B, D)
        x = x.unsqueeze(0)
        x = self.transformer(x)
        x = x.squeeze(0)
        return self.output_proj(x)

# 一个使用Transformer的Diffusion编码器
class DiffusionEncoder(nn.Module):
    def __init__(self, hidden_dim, latent_dim, T=1000):
        super(DiffusionEncoder, self).__init__()
        self.transformer_encoder = TransformerEncoder(hidden_dim, latent_dim)
        self.T = T
        self.register_buffer('betas', torch.linspace(1e-4, 0.02, T))
        self.register_buffer('alphas', 1. - self.betas)
        self.register_buffer('alphas_cumprod', torch.cumprod(self.alphas, dim=0))

    def forward(self, x):
        z_0 = self.transformer_encoder(x)
        t = torch.randint(0, self.T, (x.size(0),), device=x.device).long()
        noise = torch.randn_like(z_0)
        alpha_bar_t = self.alphas_cumprod[t].unsqueeze(1)
        z_t = torch.sqrt(alpha_bar_t) * z_0 + torch.sqrt(1 - alpha_bar_t) * noise
        return z_t, z_0

# 使用Transformer的Denoise解码器
class DenoiseDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim):
        super(DenoiseDecoder, self).__init__()
        self.transformer_decoder = TransformerDecoder(latent_dim, hidden_dim)

    def forward(self, z_t):
        return self.transformer_decoder(z_t)

# Diffusion模块
class DiffusionModule(nn.Module):
    def __init__(self, hidden_dim, latent_dim):
        super(DiffusionModule, self).__init__()
        self.encoder = DiffusionEncoder(hidden_dim, latent_dim)
        self.decoder = DenoiseDecoder(latent_dim, hidden_dim)

    def forward(self, x):
        z_t, z_0 = self.encoder(x)
        recon_x = self.decoder(z_t)
        return recon_x, z_0, z_t

# 门控机制
class GatingFusion(nn.Module):
    def __init__(self, latent_dim):
        super(GatingFusion, self).__init__()
        self.gate = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim),
            nn.Sigmoid()
        )

    def forward(self, audio_latent, text_latent):
        concat_latent = torch.cat([audio_latent, text_latent], dim=-1)
        gate_weights = self.gate(concat_latent)
        fused_latent = gate_weights * audio_latent + (1 - gate_weights) * text_latent
        return fused_latent

# 多模态融合模型
class MultimodalFusionModel(nn.Module):
    def __init__(self, hidden_dim, latent_dim, num_labels):
        super(MultimodalFusionModel, self).__init__()
        self.audio_diffusion = DiffusionModule(hidden_dim, latent_dim)
        self.text_diffusion = DiffusionModule(hidden_dim, latent_dim)
        self.gating_fusion = GatingFusion(latent_dim)
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.ReLU(),
            nn.Linear(latent_dim // 2, num_labels)
        )

    def forward(self, audio_input, text_input):
        _, audio_z, _ = self.audio_diffusion(audio_input)
        _, text_z, _ = self.text_diffusion(text_input)
        fused_z = self.gating_fusion(audio_z, text_z)
        logits = self.classifier(fused_z)
        return logits, audio_z, text_z

if __name__ == "__main__":
    batch_size = 4
    seq_length = 10
    hidden_dim = 128
    latent_dim = 64
    num_labels = 4

    audio_input = torch.randn(batch_size, seq_length, hidden_dim)
    text_input = torch.randn(batch_size, seq_length, hidden_dim)
    audio_input_pooled = audio_input.mean(dim=1)
    text_input_pooled = text_input.mean(dim=1)

    model = MultimodalFusionModel(hidden_dim, latent_dim, num_labels)
    logits, audio_z, text_z = model(audio_input_pooled, text_input_pooled)

    print("Logits shape:", logits.shape)
    print("Audio latent shape:", audio_z.shape)
    print("Text latent shape:", text_z.shape)
