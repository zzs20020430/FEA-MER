import torch
import torch.nn as nn
import torch.nn.functional as F


def get_diffusion_params(T, beta_start=1e-4, beta_end=0.02):
    betas = torch.linspace(beta_start, beta_end, T)
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    return betas, alphas, alphas_cumprod

# Transformer编码器
class TransformerVAEEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, nhead=4, num_layers=2, dropout=0.1):
        super(TransformerVAEEncoder, self).__init__()
        
        # 输入投影层
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Transformer编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 均值和方差预测层
        self.mu_layer = nn.Linear(hidden_dim, latent_dim)
        self.logvar_layer = nn.Linear(hidden_dim, latent_dim)
        
    def forward(self, x, mask=None):
        # 输入投影
        x = self.input_proj(x)
        
        # Transformer编码
        if mask is not None:
            transformer_out = self.transformer_encoder(x, src_key_padding_mask=mask)
        else:
            transformer_out = self.transformer_encoder(x)
        
        # 全局池化 - 取每个序列的平均值
        pooled = transformer_out.mean(dim=1)
        
        # 预测均值和方差
        mu = self.mu_layer(pooled)
        logvar = self.logvar_layer(pooled)
        
        # 重参数化技巧
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        
        return z, mu, logvar

# TransformerVAE解码器
class TransformerVAEDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim, nhead=4, num_layers=2, dropout=0.1):
        super(TransformerVAEDecoder, self).__init__()
        
        # 从潜在空间映射到隐藏维度
        self.latent_proj = nn.Linear(latent_dim, hidden_dim)
        
        # Transformer解码器层
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # 输出投影层
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, z, tgt_len=1):
        # 扩展潜在向量为序列
        batch_size = z.size(0)
        z_expanded = self.latent_proj(z).unsqueeze(1).expand(-1, tgt_len, -1)
        
        # 创建一个全零的目标序列
        tgt = torch.zeros_like(z_expanded)
        
        # 生成一个全1的内存掩码
        memory = z_expanded
        
        # 解码
        decoder_output = self.transformer_decoder(tgt, memory)
        
        # 输出投影
        output = self.output_proj(decoder_output)
        
        return output

# Transformer VAE模块
class TransformerVAEModule(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, nhead=4, encoder_layers=2, decoder_layers=2, dropout=0.1):
        super(TransformerVAEModule, self).__init__()
        
        self.encoder = TransformerVAEEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            nhead=nhead,
            num_layers=encoder_layers,
            dropout=dropout
        )
        
        self.decoder = TransformerVAEDecoder(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            output_dim=input_dim,
            nhead=nhead,
            num_layers=decoder_layers,
            dropout=dropout
        )
        
    def forward(self, x, mask=None, tgt_len=1):
        # 编码
        z, mu, logvar = self.encoder(x, mask)
        
        # 解码
        recon_x = self.decoder(z, tgt_len)
        
        return recon_x, z, mu, logvar

# TransformerVAE融合模型
class TransformerVAEFusionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, num_labels, nhead=4, num_layers=2, dropout=0.1):
        super(TransformerVAEFusionModel, self).__init__()
        
        # 音频和文本各自的VAE
        self.audio_vae = TransformerVAEModule(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            nhead=nhead,
            encoder_layers=num_layers,
            decoder_layers=num_layers,
            dropout=dropout
        )
        
        self.text_vae = TransformerVAEModule(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            nhead=nhead,
            encoder_layers=num_layers,
            decoder_layers=num_layers,
            dropout=dropout
        )
        
        # 注意力融合模块
        self.fusion_attn = nn.MultiheadAttention(
            embed_dim=latent_dim,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )
        
        # 最终分类头
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim // 2, num_labels)
        )
        
    def forward(self, audio_input, text_input, audio_mask=None, text_mask=None):
        # 编码音频
        _, audio_z, audio_mu, audio_logvar = self.audio_vae(audio_input, audio_mask)
        
        # 编码文本
        _, text_z, text_mu, text_logvar = self.text_vae(text_input, text_mask)
        
        # 注意力融合
        audio_z_seq = audio_z.unsqueeze(1)  # [B, 1, D]
        text_z_seq = text_z.unsqueeze(1)    # [B, 1, D]
        
        # 将音频作为查询，文本作为键和值
        fused_audio, _ = self.fusion_attn(audio_z_seq, text_z_seq, text_z_seq)
        # 将文本作为查询，音频作为键和值
        fused_text, _ = self.fusion_attn(text_z_seq, audio_z_seq, audio_z_seq)
        
        # 融合表示
        fused_z = (fused_audio.squeeze(1) + fused_text.squeeze(1)) / 2
        
        # 分类
        logits = self.classifier(fused_z)
        
        return logits, audio_z, text_z, fused_z, audio_mu, audio_logvar, text_mu, text_logvar
    
    def compute_kl_loss(self, mu, logvar):
        # KL散度损失: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return kl_loss

# 一个简化的Diffusion编码器（取代原来的VAE）
class DiffusionEncoder(nn.Module):
    def __init__(self, hidden_dim, latent_dim, T=1000):
        super(DiffusionEncoder, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        self.T = T
        self.register_buffer('betas', torch.linspace(1e-4, 0.02, T))
        self.register_buffer('alphas', 1. - self.betas)
        self.register_buffer('alphas_cumprod', torch.cumprod(self.alphas, dim=0))

    def forward(self, x):
        z_0 = self.fc(x)
        t = torch.randint(0, self.T, (x.size(0),), device=x.device).long()
        noise = torch.randn_like(z_0)
        alpha_bar_t = self.alphas_cumprod[t].unsqueeze(1)
        z_t = torch.sqrt(alpha_bar_t) * z_0 + torch.sqrt(1 - alpha_bar_t) * noise
        return z_t, z_0

# 简化的去噪器（作为解码器）
class DenoiseDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim):
        super(DenoiseDecoder, self).__init__()
        self.denoiser = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, z_t):
        return self.denoiser(z_t)

# 真实Diffusion机制的替代VAE结构
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

# 完整模态融合模型定义
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
        # 分别经过扩散模块
        _, audio_z, _ = self.audio_diffusion(audio_input)
        _, text_z, _ = self.text_diffusion(text_input)

        # 融合两个模态的z
        fused_z = self.gating_fusion(audio_z, text_z)

        # 分类
        logits = self.classifier(fused_z)
        return logits, audio_z, text_z

# 示例用法
if __name__ == "__main__":
    batch_size = 4
    seq_length = 10
    hidden_dim = 128
    latent_dim = 64
    num_labels = 4

    # 随机生成样本
    audio_input = torch.randn(batch_size, seq_length, hidden_dim)
    text_input = torch.randn(batch_size, seq_length, hidden_dim)

    # 平均池化序列维度到单一向量
    audio_input_pooled = audio_input.mean(dim=1)
    text_input_pooled = text_input.mean(dim=1)

    model = MultimodalFusionModel(hidden_dim, latent_dim, num_labels)
    logits, audio_z, text_z = model(audio_input_pooled, text_input_pooled)

    print("Logits shape:", logits.shape)
    print("Audio latent shape:", audio_z.shape)
    print("Text latent shape:", text_z.shape)
