import torch
import torch.nn as nn
from transformers import BertConfig, BertModel


class QueryMultimodalBertModel(nn.Module):
    def __init__(self, config):
        super(QueryMultimodalBertModel, self).__init__()
        self.config = config
        self.bert = BertModel(config)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.max_position_embeddings = config.max_position_embeddings  # 最大位置编码长度
        
        # 创建可学习的查询向量，为每个分类标签创建一个查询向量
        self.class_queries = nn.Parameter(
            torch.randn(config.num_labels, config.hidden_size)
        )
        
        # 初始化查询向量
        nn.init.xavier_uniform_(self.class_queries)
        
        # 保存当前注意力掩码和钩子
        self.current_attention_mask = None
        self.hooks = []

    def create_query_attention_mask(self, batch_size, seq_len, num_queries, device):
        """
        创建注意力掩码，防止查询向量之间相互关注
        
        Args:
            batch_size: 批次大小
            seq_len: 序列总长度
            num_queries: 查询向量数量
            device: 设备
            
        Returns:
            attention_mask: 注意力掩码矩阵 [batch_size, seq_len, seq_len]
        """
        # 创建初始掩码矩阵，值全为1，表示所有位置都可以互相关注
        attention_mask = torch.ones((batch_size, seq_len, seq_len), device=device)
        
        # 查询向量的起始位置
        query_start = seq_len - num_queries
        
        # 设置查询区域的掩码，使查询向量之间不能互相关注
        # 除对角线外都设为0
        for b in range(batch_size):
            attention_mask[b, query_start:, query_start:] = torch.eye(num_queries, device=device)
            
        return attention_mask
    
    def register_attention_hooks(self):
        """注册BERT自注意力层的钩子，用于修改注意力分数"""
        # 先移除现有的钩子
        self.remove_hooks()
        
        # 定义注意力分数修改钩子
        def hook_fn(module, input_args, output):
            # 检查我们是否有注意力掩码
            if self.current_attention_mask is None:
                return output
                
            # 输出是一个元组，第一个元素是注意力分数
            attention_scores = output[0]  # [batch_size, num_heads, seq_len, seq_len]
            
            # 检查形状是否匹配
            batch_size, num_heads, seq_len, _ = attention_scores.shape
            if seq_len != self.current_attention_mask.size(1):
                return output
                
            # 调整掩码形状以匹配注意力头
            expanded_mask = self.current_attention_mask.unsqueeze(1).expand(
                -1, num_heads, -1, -1
            )
            
            # 将掩码应用到注意力分数上
            # 掩码为0的地方会被设置为一个很大的负数，让softmax后趋近于0
            masked_scores = torch.where(
                expanded_mask == 0,
                torch.tensor(-10000.0, device=attention_scores.device),
                attention_scores
            )
            
            # 返回修改后的注意力分数和其他输出
            return (masked_scores,) + output[1:]
        
        # 遍历所有BERT自注意力层并注册钩子
        for name, module in self.bert.named_modules():
            if "attention.self" in name:
                hook = module.register_forward_hook(hook_fn)
                self.hooks.append(hook)
    
    def remove_hooks(self):
        """移除所有注册的钩子"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def forward(self, audio_input, text_input):
        """
        audio_input: [batch_size, seq_audio_len, hidden_size] - 已处理的音频序列嵌入
        text_input: [batch_size, seq_text_len, hidden_size] - 已处理的文本序列嵌入
        """
        device = audio_input.device
        self.bert.to(device)
        self.classifier.to(device)

        # 获取批次大小
        batch_size = audio_input.size(0)

        # 创建 [CLS] 和 [SEP] token 的嵌入
        cls_token = torch.zeros((batch_size, 1, audio_input.size(-1))).to(device)
        sep_token = torch.zeros((batch_size, 1, audio_input.size(-1))).to(device)

        # 拼接音频和文本序列： [CLS] + audio_input + [SEP] + text_input + [SEP]
        audio_input = torch.cat([cls_token, audio_input, sep_token], dim=1).to(device)
        text_input = torch.cat([text_input, sep_token], dim=1).to(device)
        
        # 使用sum检测填充的零值
        # 检测音频输入中的填充（将每个token的所有特征维度求和，如果为0则是填充token）
        audio_padding_mask = (torch.sum(torch.abs(audio_input), dim=-1) != 0).float()  # [batch_size, audio_seq_len]
        
        # 检测文本输入中的填充
        text_padding_mask = (torch.sum(torch.abs(text_input), dim=-1) != 0).float()  # [batch_size, text_seq_len]

        # 拼接后的输入序列
        inputs_embeds = torch.cat([audio_input, text_input], dim=1).to(device)
        
        # 拼接填充掩码
        padding_mask = torch.cat([audio_padding_mask, text_padding_mask], dim=1)  # [batch_size, audio_seq_len + text_seq_len]
        
        # 将查询向量扩展到批次维度并添加到序列末尾
        # [num_labels, hidden_size] -> [batch_size, num_labels, hidden_size]
        expanded_queries = self.class_queries.unsqueeze(0).expand(batch_size, -1, -1)
        
        # 拼接查询向量到输入序列: [batch_size, seq_len + num_labels, hidden_size]
        concat_inputs = torch.cat([inputs_embeds, expanded_queries], dim=1).to(device)
        
        # 查询向量不是填充，为其掩码添加全1
        query_padding_mask = torch.ones((batch_size, self.config.num_labels), device=device)
        
        # 更新填充掩码以包含查询向量
        full_padding_mask = torch.cat([padding_mask, query_padding_mask], dim=1)  # [batch_size, total_seq_len]

        # 检查序列长度是否超出最大限制
        if concat_inputs.size(1) > self.max_position_embeddings:
            # 保留查询向量，截断中间部分的序列
            query_part = concat_inputs[:, -self.config.num_labels:]
            other_part = concat_inputs[:, :self.max_position_embeddings-self.config.num_labels]
            concat_inputs = torch.cat([other_part, query_part], dim=1)
            
            # 同样处理填充掩码
            query_mask_part = full_padding_mask[:, -self.config.num_labels:]
            other_mask_part = full_padding_mask[:, :self.max_position_embeddings-self.config.num_labels]
            full_padding_mask = torch.cat([other_mask_part, query_mask_part], dim=1)

        # 创建查询掩码，防止查询向量之间互相关注
        seq_len = concat_inputs.size(1)
        query_attention_mask = self.create_query_attention_mask(
            batch_size, seq_len, self.config.num_labels, device
        )
        
        # 应用填充掩码：如果token i或j是填充位置，则不互相关注
        for b in range(batch_size):
            for i in range(seq_len):
                for j in range(seq_len):
                    if full_padding_mask[b, i] == 0 or full_padding_mask[b, j] == 0:
                        query_attention_mask[b, i, j] = 0
        
        # 设置当前注意力掩码并注册钩子
        self.current_attention_mask = query_attention_mask
        self.register_attention_hooks()
        
        # 将拼接后的序列传入BERT模型
        outputs = self.bert(
            inputs_embeds=concat_inputs,
            attention_mask=full_padding_mask
        )
        
        # 移除钩子，避免影响后续计算
        self.remove_hooks()
        self.current_attention_mask = None
        
        # 获取所有查询向量的输出
        query_outputs = outputs.last_hidden_state[:, -self.config.num_labels:]
        
        # 获取CLS输出作为融合特征
        cls_output = outputs.last_hidden_state[:, 0, :]
        
        # 计算每个查询向量的得分（使用点积）并输出为logits
        logits = torch.bmm(
            query_outputs, 
            cls_output.unsqueeze(-1)
        ).squeeze(-1)

        return logits, cls_output  # 返回分类logits和融合特征 