import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2Processor
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2Model, Wav2Vec2PreTrainedModel
from model.network.audio_encoder import AudioModel
from model.network.regression_head import RegressionHead
import math
from typing import Optional, List, Union, Tuple

# LoRA层实现
class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, r=8, alpha=16):
        super().__init__()
        self.orig_module = nn.Linear(in_features, out_features, bias=True)
        self.r = r
        self.alpha = alpha
        
        # 固定原始层的参数
        for param in self.orig_module.parameters():
            param.requires_grad = False
            
        # 低秩分解层
        self.lora_A = nn.Parameter(torch.zeros(in_features, r))
        self.lora_B = nn.Parameter(torch.zeros(r, out_features))
        
        # 初始化
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        
        self.scaling = alpha / r
        
    def forward(self, x):
        # 原始前向传播
        orig_output = self.orig_module(x)
        
        # LoRA前向传播
        lora_output = (x @ self.lora_A) @ self.lora_B
        
        # 缩放并组合
        return orig_output + lora_output * self.scaling

# 修改后的Wav2Vec2LoRA模型
class Wav2Vec2LoRAModel(Wav2Vec2Model):
    def __init__(self, config, r=8, alpha=16, target_modules=None):
        super().__init__(config)
        self.r = r
        self.alpha = alpha
        
        # 默认要应用LoRA的模块列表
        if target_modules is None:
            self.target_modules = [
                "encoder.layers.{}.attention.q_proj",
                "encoder.layers.{}.attention.k_proj",
                "encoder.layers.{}.attention.v_proj",
                "encoder.layers.{}.feed_forward.intermediate_dense",
                "encoder.layers.{}.feed_forward.output_dense"
            ]
        else:
            self.target_modules = target_modules
            
        # 应用LoRA
        self._apply_lora()
    
    def _apply_lora(self):
        # 遍历所有需要应用LoRA的层
        for module_name in self.target_modules:
            for i in range(len(self.encoder.layers)):
                # 格式化模块名称以包含层索引
                formatted_name = module_name.format(i)
                
                # 解析模块路径
                parts = formatted_name.split('.')
                parent = self
                for part in parts[:-1]:
                    if part.isdigit():
                        parent = parent[int(part)]
                    else:
                        parent = getattr(parent, part)
                
                # 获取最后一个属性名
                final_part = parts[-1]
                
                # 获取原始线性层
                original_layer = getattr(parent, final_part)
                
                if isinstance(original_layer, nn.Linear):
                    # 创建LoRA层并替换原始层
                    lora_layer = LoRALinear(
                        original_layer.in_features,
                        original_layer.out_features,
                        r=self.r,
                        alpha=self.alpha
                    )
                    
                    # 复制原始层的权重和偏置
                    lora_layer.orig_module.weight.data = original_layer.weight.data
                    if original_layer.bias is not None:
                        lora_layer.orig_module.bias.data = original_layer.bias.data
                    
                    # 替换原始层
                    setattr(parent, final_part, lora_layer)

class AudioModelLoRA(Wav2Vec2PreTrainedModel):
    r"""Speech emotion classifier with LoRA fine-tuning."""
    def __init__(self, config, r=8, alpha=16):
        super().__init__(config)
        self.config = config
        self.wav2vec2 = Wav2Vec2LoRAModel(config, r=r, alpha=alpha)
        self.classifier = RegressionHead(config)
        self.init_weights()
        self.is_emotion = False
        
    def forward(self, input_values, attention_mask=None):
        outputs = self.wav2vec2(input_values, attention_mask=attention_mask)
        hidden_states0 = outputs[0]
        hidden_states1 = torch.mean(hidden_states0, dim=1)
        
        if not self.is_emotion:
            return hidden_states0, hidden_states1, self.generate_feature_attention_mask(hidden_states0)
        else:
            return self.classifier(hidden_states1)

    def generate_feature_attention_mask(self, hidden_states):
        # 检查特征张量中的零填充部分
        # 假设特征张量中的零填充部分在最后一个维度上为全零
        # 这里我们检查每个序列的第一个维度是否有非零值
        feature_attention_mask = (hidden_states.sum(dim=-1) != 0).long()
        return feature_attention_mask
    
    def get_trainable_parameters(self):
        """获取可训练的参数（只有LoRA参数）"""
        trainable_params = []
        for name, param in self.named_parameters():
            if "lora_A" in name or "lora_B" in name or "classifier" in name:
                trainable_params.append(param)
            else:
                param.requires_grad = False
        return trainable_params
    
class AudioEncoderLoRA(nn.Module):
    r"""Speech emotion classifier with LoRA."""
    def __init__(self, processor=None, audio_model=None, sample_rate=16000, model_path=None, 
                 is_emotion=False, r=8, alpha=16):
        super(AudioEncoderLoRA, self).__init__()
        
        if model_path is None:
            self.processor = processor
            if isinstance(audio_model, AudioModel):
                # 将标准AudioModel转换为LoRA版本
                config = audio_model.config
                self.audio_model = AudioModelLoRA(config, r=r, alpha=alpha)
                # 复制权重
                self._copy_weights(audio_model, self.audio_model)
            else:
                self.audio_model = audio_model
        else:
            self.processor = Wav2Vec2Processor.from_pretrained(model_path)
            self.audio_model = AudioModelLoRA.from_pretrained(model_path, r=r, alpha=alpha)
        
        self.sample_rate = sample_rate
        self.audio_model.is_emotion = is_emotion

        # 如果是情感编码器，冻结除了LoRA和分类头以外的所有参数
        if is_emotion:
            for name, param in self.audio_model.named_parameters():
                if "lora_A" not in name and "lora_B" not in name and "classifier" not in name:
                    param.requires_grad = False

    def _copy_weights(self, src_model, dst_model):
        """将源模型的权重复制到目标模型"""
        # 复制wav2vec2的权重
        for src_name, src_param in src_model.wav2vec2.named_parameters():
            if "lora" not in src_name:  # 跳过LoRA参数
                for dst_name, dst_param in dst_model.wav2vec2.named_parameters():
                    if dst_name == src_name:
                        dst_param.data.copy_(src_param.data)
                        break
        
        # 复制分类器的权重
        try:
            dst_model.classifier.load_state_dict(src_model.classifier.state_dict())
        except:
            print("Warning: Could not copy classifier weights.")

    def forward(self, audio_inputs):
        inputs = self.processor(audio_inputs, sampling_rate=self.sample_rate, 
                               return_attention_mask=True, padding=True, return_tensors='pt')
        input_values = inputs['input_values']
        attention_mask = inputs['attention_mask']
        return self.audio_model(input_values.cuda(), attention_mask.cuda())
    
    def get_trainable_parameters(self):
        """获取可训练的参数（只有LoRA参数和分类器）"""
        return self.audio_model.get_trainable_parameters()
