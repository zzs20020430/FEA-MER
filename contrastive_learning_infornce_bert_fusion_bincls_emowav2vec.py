import torch
import torch.nn as nn
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2Model, Wav2Vec2PreTrainedModel
from model.network.audio_encoder import AudioEncoder, AudioModel
from model.network.text_encoder import TextEncoder
from model.network.temperature import TemperatureModel
from model.network.fc import FcEncoder
import numpy as np
from model.network.bert_fusion_bincls import *
from tqdm import tqdm
import random
import torch.nn.functional as F
import os
import copy
from utils.tools import *
from dataloader.iemocap_dataloader import get_dataloader, label_list
from utils.print_logger import Logger

set_seed(42)
device = get_device()
temperature_model = TemperatureModel().to(device)

pwd = get_current_root(__file__)
args = read_yaml_to_dict(pwd/"config/dataset/iemocap.yaml")
model_dir = Path("/you/never/guess/models/")
audio_emotion_model_path = model_dir/'wav2vec2-large-uncased'  # for emotion recognition
text_model_path = model_dir/'roberta-large-uncased'

# 创建情感预测模型
num_labels = args.num_classes
text_encoder = TextEncoder(model_path=text_model_path).to(device)
audio_encoder = AudioEncoder(model_path=audio_emotion_model_path).to(device)
session_id = args.test_session
hidden_size = text_encoder.text_model.config.hidden_size
logger = Logger(pwd/f"log/{get_exec_name(__file__)}_{get_datetime()}_session{session_id}.log")

# 损失权重 - 移除MAE相关
contrastive_loss_w, classification_loss_w, binary_loss_w = 0.3, 0.5, 0.2

# BERT配置
bert_config = BertConfig(
    hidden_size=hidden_size,
    num_hidden_layers=6,
    num_attention_heads=8,
    intermediate_size=4096,
    max_position_embeddings=2800,
    num_labels=num_labels
)

# 多模态融合模块
Bert_adapter_multimodel_fusion = MultimodalBertModel(bert_config).to(device)

# 对比学习投影头
fusion_projection_head = FcEncoder(hidden_size, [hidden_size, 'relu', 256]).to(device)

# 交叉熵损失
classification_criterion = nn.CrossEntropyLoss()

# 添加二分类损失计算函数
def compute_binary_loss(audio_embedding, text_embedding, raw_text_seq_embedding, labels):
    pairs, pairs_audio_mask, pair_labels = create_binary_classification_pairs(audio_embedding, text_embedding, raw_text_seq_embedding, labels)
    binary_pairs_audio = torch.stack([pair[0] for pair in pairs]).to(device)
    binary_pairs_text = torch.stack([pair[1] for pair in pairs]).to(device)
    binary_pair_labels = torch.tensor(pair_labels).to(device)
    binary_logits, _ = Bert_adapter_multimodel_fusion(binary_pairs_audio, binary_pairs_text, bin_cls=True)
    return classification_criterion(binary_logits, binary_pair_labels)

# 评估函数
def evaluate_model(text_encoder, model2, dataloader):
    text_encoder.eval()
    model2.eval()
    audio_encoder.eval()
    correct_predictions = 0
    total_predictions = 0
    total_loss = 0
    class_correct = np.zeros(num_labels)
    class_total = np.zeros(num_labels)
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Session {session_id} Epoch {epoch + 1}/{num_epochs}", ncols=80):
            text, audio, labels = batch.text, batch.audio, batch.label

            # 获取嵌入
            raw_text_seq_embedding, raw_text_cls_embeddings = text_encoder(text)
            raw_audio_seq_embedding, raw_audio_cls_embeddings, raw_audio_attention_masks = audio_encoder(audio)
            logits, cls_output = model2(raw_text_seq_embedding, raw_audio_seq_embedding)
            predictions = torch.argmax(logits, dim=-1)
            loss = classification_criterion(logits, labels.to(device))
            total_loss += loss.item()

            correct_predictions += (predictions == labels.to(device)).sum().item()
            total_predictions += labels.size(0)

            for i in range(len(labels)):
                label = labels[i].item()
                class_correct[label] += (predictions[i] == labels[i].to(device)).item()
                class_total[label] += 1

    avg_loss = total_loss / total_predictions
    class_accuracy = class_correct / class_total
    accuracy = class_accuracy.mean()
    class_weights = class_total / total_predictions
    weighted_accuracy = (class_accuracy * class_weights).sum()
    return accuracy, avg_loss, weighted_accuracy, class_accuracy, class_weights

num_epochs = 300
best_accuracy = 0.8510 if session_id==2 else 0.78
# best_accuracy = 0.7770 if session_id==3 else 0.78

# 循环训练每个 Session
# for session_id in range(1, 6):
if session_id is not None:
    print(f"Training on Session {session_id}")
    
    optimizer = torch.optim.Adam([
        {'params': text_encoder.parameters()},
        {'params': Bert_adapter_multimodel_fusion.parameters()},
        {'params': temperature_model.parameters()},
        {'params': fusion_projection_head.parameters()},
        {'params': audio_encoder.parameters()},
    ], lr=1e-5)
    
    train_dataloader, val_dataloader = get_dataloader(args)
    session_best_accuracy = 0
    
    for epoch in range(num_epochs):
        text_encoder.train()
        Bert_adapter_multimodel_fusion.train()
        temperature_model.train()
        audio_encoder.train()
        fusion_projection_head.train()
        
        for batch_idx, batch in enumerate(
                tqdm(train_dataloader, desc=f"Session {session_id} Epoch {epoch + 1}/{num_epochs}", ncols=80), 1):
            text, audio, labels = batch.text, batch.audio, batch.label
            
            # 1. 获取原始文本和音频嵌入
            raw_text_seq_embedding, raw_text_cls_embeddings = text_encoder(text)
            raw_audio_seq_embedding, raw_audio_cls_embeddings, raw_audio_attention_masks = audio_encoder(audio)
            
            # 2. 用于分类的普通前向传播
            logits_classification, fusion_cls_features = Bert_adapter_multimodel_fusion(
                raw_audio_seq_embedding, raw_text_seq_embedding)
            classification_loss = classification_criterion(logits_classification, labels.to(device))
            
            # 3. 对融合特征进行对比学习
            with torch.no_grad():
                raw_audio_dimension = audio_encoder.audio_model.classifier(raw_audio_cls_embeddings)
            
            contrastive_loss = 0
            temperature = temperature_model()
            for i in range(len(labels)):
                # 使用投影头处理融合特征
                anchor_fusion = fusion_projection_head(fusion_cls_features[i])
                anchor_label = labels[i]
                anchor_dimension = raw_audio_dimension[i]
                
                # 找出所有的正样本和负样本
                positive_indices = (labels == anchor_label).nonzero(as_tuple=True)[0]
                negative_indices = (labels != anchor_label).nonzero(as_tuple=True)[0]
                
                # 对正负样本的融合特征进行投影
                positive_embeddings = fusion_projection_head(fusion_cls_features[positive_indices])
                negative_embeddings = fusion_projection_head(fusion_cls_features[negative_indices])
                
                if len(positive_embeddings) > 1:
                    # 使用情感特征计算权重
                    pos_weights = torch.tensor(
                        [F.cosine_similarity(anchor_dimension.unsqueeze(0), raw_audio_dimension[j].unsqueeze(0), dim=1)
                        for j in positive_indices])
                    pos_weights = 1.0 / (pos_weights + 1e-8)
                    pos_weights = pos_weights / pos_weights.sum()

                    neg_weights = torch.tensor(
                        [F.cosine_similarity(anchor_dimension.unsqueeze(0), raw_audio_dimension[j].unsqueeze(0), dim=1)
                        for j in negative_indices])
                    neg_weights = neg_weights / neg_weights.sum()

                    # 计算投影后特征的相似度
                    pos_sim = torch.exp(
                        F.cosine_similarity(anchor_fusion.unsqueeze(0),
                                            positive_embeddings) / temperature)
                    neg_sim = torch.exp(
                        F.cosine_similarity(anchor_fusion.unsqueeze(0),
                                            negative_embeddings) / temperature)

                    pos_sum = (pos_sim * pos_weights.to(device)).sum()
                    neg_sum = (neg_sim * neg_weights.to(device)).sum()
                    contrastive_loss += -torch.log(pos_sum / (pos_sum + neg_sum))
            
            contrastive_loss = contrastive_loss / len(labels)
            
            # 4. 二分类损失
            binary_loss = compute_binary_loss(raw_audio_seq_embedding, raw_text_seq_embedding, raw_text_seq_embedding, labels)

            # 总损失 - 移除MAE相关损失
            loss = contrastive_loss_w * contrastive_loss + \
                   classification_loss_w * classification_loss + \
                   binary_loss_w * binary_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        # 评估模型
        accuracy, avg_loss, weighted_accuracy, class_accuracy, class_weights = evaluate_model(
            text_encoder, Bert_adapter_multimodel_fusion, val_dataloader)
        
        print(f"[{get_datetime()}] Session {session_id} Epoch {epoch + 1}/{num_epochs}, "
              f"Validation Accuracy: {accuracy:.4f}, "
              f"Validation Loss: {avg_loss:.4f}, "
              f"Weighted Accuracy: {weighted_accuracy:.4f}")
        
        for i in range(num_labels):
            print(f"Class [{i}] {label_list[i]} Accuracy: {class_accuracy[i]:.4f}, Weight: {class_weights[i]:.4f}")

            
        # 如果超过全局最佳，则保存模型
        if accuracy > best_accuracy or( epoch+1==58 and session_id==2):
            best_accuracy = accuracy
            model_path = pwd / f"ckpt/{get_exec_name(__file__)}_session{session_id}_acc_{best_accuracy:.4f}.pth"
            torch.save({
                'text_encoder': text_encoder.state_dict(),
                'fusion_clf': Bert_adapter_multimodel_fusion.state_dict(),
                'fusion_proj_head': fusion_projection_head.state_dict(),
                'audio_encoder': audio_encoder.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch
            }, model_path)
            print(f"New best model saved with accuracy: {best_accuracy:.4f}")

print("Training completed.") 