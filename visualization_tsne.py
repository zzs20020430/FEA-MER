import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from pathlib import Path
import argparse
from model.network.audio_encoder import AudioEncoder
from model.network.text_encoder import TextEncoder
from model.network.bert_fusion_bincls import *
from dataloader.iemocap_dataloader import get_dataloader, label_list
from utils.tools import get_device, get_current_root, read_yaml_to_dict, choose_gpu
import seaborn as sns
from tqdm import tqdm
import os
import pickle

# Set device
os.environ['CUDA_VISIBLE_DEVICES'] = choose_gpu()
# device = "cpu"
device = get_device()

# Load configuration
pwd = Path(os.getcwd())
args = read_yaml_to_dict(pwd/"config/dataset/iemocap.yaml")

# Model paths
model_dir = Path("/18t/data/home/wangchai/SpeechEmotion/models/")
audio_emotion_model_path = model_dir/'wav2vec2-large-uncased'
text_model_path = model_dir/'roberta-large-uncased'
output_dir=pwd/"visualization_results"

def init_models():
    # Initialize encoders
    text_encoder = TextEncoder(model_path=text_model_path).to(device)
    audio_encoder = AudioEncoder(model_path=audio_emotion_model_path).to(device)

    # 创建情感预测模型
    num_labels = args.num_classes
    hidden_size = text_encoder.text_model.config.hidden_size
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
    return text_encoder,audio_encoder,Bert_adapter_multimodel_fusion

c=['r', 'g', 'b', 'gray', 'm', 'y', 'k', 'w']#ang hap sad neu
stages=['raw','finetuned']
# stages=['raw']
session_ckpt={
    1:"/18t/data/home/zzs/proj/Emotion_zzs/MEIFDN_early_fusion/ckpt/contrastive_learning_infornce_bert_fusion_bincls_emowav2vec_session1_acc_0.7953.pth",
    2:"/18t/data/home/zzs/proj/Emotion_zzs/MEIFDN_early_fusion/ckpt/contrastive_learning_infornce_bert_fusion_bincls_emowav2vec_session2_acc_0.8514.pth",
    3:"",
    4:"/18t/data/home/zzs/proj/Emotion_zzs/MEIFDN_early_fusion/ckpt/contrastive_learning_infornce_bert_fusion_bincls_emowav2vec_session4_acc_0.7894.pth",
    5:"/18t/data/home/zzs/proj/Emotion_zzs/MEIFDN_early_fusion/ckpt/contrastive_learning_infornce_bert_fusion_bincls_emowav2vec_session5_acc_0.7986.pth",
}

def extract_features(dataloader, stage:str, fusion, model_path, text_encoder, audio_encoder, Bert_adapter_multimodel_fusion):
    """Extract raw features from the encoders"""
    text_encoder.eval()
    audio_encoder.eval()
    Bert_adapter_multimodel_fusion.eval()

    if stage in ['finetuned']:
        checkpoint = torch.load(model_path, map_location=device)
        text_encoder.load_state_dict(checkpoint['text_encoder'])
        audio_encoder.load_state_dict(checkpoint['audio_encoder'])
        Bert_adapter_multimodel_fusion.load_state_dict(checkpoint['fusion_clf'])

    if fusion:
        all_fusion_cls=[]
    else:
        all_text_features = []
        all_audio_features = []

    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Extracting {stage} features", ncols=75):
            text, audio, labels = batch.text, batch.audio, batch.label
            if fusion:
                all_fusion_cls.append(Bert_adapter_multimodel_fusion(text_encoder(text)[0],audio_encoder(audio)[0])[1].cpu())
            else:
                # Get text features
                _, text_cls_embeddings = text_encoder(text)
                all_text_features.append(text_cls_embeddings.cpu())
                
                # Get audio features
                _, audio_cls_embeddings, _ = audio_encoder(audio)
                all_audio_features.append(audio_cls_embeddings.cpu())
                
            all_labels.append(labels.cpu())
    return (
        torch.cat(all_fusion_cls, dim=0),
        torch.cat(all_labels, dim=0)
    )if fusion else(
        torch.cat(all_text_features, dim=0),
        torch.cat(all_audio_features, dim=0),
        torch.cat(all_labels, dim=0)
    )

def extract_finetuned_features(dataloader, model_path, fusion=False):
    """Extract fine-tuned features from the encoders"""
    # Load fine-tuned model
    checkpoint = torch.load(model_path, map_location=device)
    text_encoder.load_state_dict(checkpoint['text_encoder'])
    audio_encoder.load_state_dict(checkpoint['audio_encoder'])
    if fusion:
        Bert_adapter_multimodel_fusion.load_state_dict(checkpoint['fusion_clf'])
        all_fusion_cls=[]
        Bert_adapter_multimodel_fusion.eval()
    else:
        all_text_features = []
        all_audio_features = []
    text_encoder.eval()
    audio_encoder.eval()
    
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting raw features"):
            text, audio, labels = batch.text, batch.audio, batch.label
            
            if fusion:
                all_fusion_cls.append(Bert_adapter_multimodel_fusion(text_encoder(text)[0],audio_encoder(audio)[0])[1].cpu())
            else:
                # Get text features
                _, text_cls_embeddings = text_encoder(text)
                all_text_features.append(text_cls_embeddings.cpu())
                
                # Get audio features
                _, audio_cls_embeddings, _ = audio_encoder(audio)
                all_audio_features.append(audio_cls_embeddings.cpu())
                
            all_labels.append(labels.cpu())
    
    return (
        torch.cat(all_fusion_cls, dim=0),
        torch.cat(all_labels, dim=0)
    )if fusion else(
        torch.cat(all_text_features, dim=0),
        torch.cat(all_audio_features, dim=0),
        torch.cat(all_labels, dim=0)
    )

def plot_tsne(features, labels, title, output_path=None):
    font_size=24
    """Plot t-SNE visualization of features"""
    # Convert to numpy arrays
    features = features.numpy()
    labels = labels.numpy()

    # Perform t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    features_2d = tsne.fit_transform(features)
    # https://github.com/matplotlib/matplotlib/issues/16616

    # Create plot
    plt.figure(figsize=(10, 8))
    
    # Create custom colormap from the predefined colors
    import matplotlib.colors
    cmap, norm = matplotlib.colors.from_levels_and_colors(
        np.arange(args.num_classes+1), c[:args.num_classes]
    )
    
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=labels, cmap=cmap, norm=norm, alpha=0.5)
    
    # Add legend
    plt.legend(handles=scatter.legend_elements()[0], labels=label_list[:args.num_classes],prop={'family' : 'Times New Roman', 'size'   : font_size})
    
    plt.title(title, fontdict={'family' : 'Times New Roman', 'size'   : font_size})
    plt.xlabel('t-SNE Dimension 1', fontdict={'family' : 'Times New Roman', 'size'   : font_size})
    plt.ylabel('t-SNE Dimension 2', fontdict={'family' : 'Times New Roman', 'size'   : font_size})
    plt.yticks(fontproperties = 'Times New Roman', size = 18)
    plt.xticks(fontproperties = 'Times New Roman', size = 18)
    if output_path:
        plt.savefig(output_path)
    plt.show()

def compare_raw_vs_finetuned(raw_features, finetuned_features, labels, output_dir):
    """Compare raw and fine-tuned features using t-SNE"""
    # Convert to numpy arrays
    raw_features = raw_features.numpy()
    finetuned_features = finetuned_features.numpy()
    labels = labels.numpy()
    
    # Perform t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    raw_features_2d = tsne.fit_transform(raw_features)
    finetuned_features_2d = tsne.fit_transform(finetuned_features)
    
    # Create plot
    plt.figure(figsize=(15, 6))
    
    # Plot raw features
    plt.subplot(1, 2, 1)
    scatter1 = plt.scatter(raw_features_2d[:, 0], raw_features_2d[:, 1], c=labels, cmap='viridis', alpha=0.6, marker='o')
    plt.title('Raw Features')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    
    # Plot fine-tuned features
    plt.subplot(1, 2, 2)
    scatter2 = plt.scatter(finetuned_features_2d[:, 0], finetuned_features_2d[:, 1], c=labels, cmap='viridis', alpha=0.6, marker='^')
    plt.title('Fine-tuned Features')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    
    # Add legend
    plt.legend(handles=scatter1.legend_elements()[0], labels=label_list[:args.num_classes])
    
    plt.tight_layout()
    
    # Save plot
    output_path = Path(output_dir) / 'raw_vs_finetuned_comparison.png'
    plt.savefig(output_path)
    plt.show()

if __name__ == "__main__":
    fusion=True
    features={}
    if fusion:
        features['fusion_cls']={stage:[] for stage in stages}
    else:
        features['text'] = {stage:[] for stage in stages}
        features['audio'] = {stage:[] for stage in stages}
    labels=[]
    for session in range(2,3):
        if session==3:
            continue
        args.test_session=session
        _, val_dataloader = get_dataloader(args)
        text_encoder, audio_encoder, Bert_adapter_multimodel_fusion=init_models()
        loaded=False
        for stage in stages:
            label_path=output_dir/f'cache_earlyfusion'/f"session{session}_{stage}_label.pkl"
            feat_path=output_dir/f'cache_earlyfusion'/f"session{session}_{stage}_fusion_feature.pkl"
            text_feat_path=output_dir/f'cache_earlyfusion'/f"session{session}_{stage}_text_feature.pkl"
            audio_feat_path=output_dir/f'cache_earlyfusion'/f"session{session}_{stage}_audio_feature.pkl"            
            if not (Path.exists(label_path)or (Path.exists(feat_path) and fusion) or (Path.exists(text_feat_path)and Path.exists(audio_feat_path))):
                features[stage] = extract_features(val_dataloader,stage ,fusion, session_ckpt[session], text_encoder, audio_encoder, Bert_adapter_multimodel_fusion)
                if fusion:
                    features['fusion_cls'][stage].append(features[stage][0])
                    with open(feat_path,'wb')as f:
                        pickle.dump(features[stage][0],f)
                else:
                    features['text'][stage].append(features[stage][0])
                    features['audio'][stage].append(features[stage][1])
                    with open(text_feat_path,'wb')as f:
                        pickle.dump(features[stage][0],f)
                    with open(audio_feat_path,'wb')as f:
                        pickle.dump(features[stage][1],f)
                if not loaded:
                    loaded=True
                    labels.append(features[stage][-1])
                with open(label_path,'wb')as f:
                    pickle.dump(features[stage][-1],f)
            else:
                if fusion:
                    with open(feat_path,'rb')as f:
                        features['fusion_cls'][stage].append(pickle.load(f))
                else:
                    with open(text_feat_path,'rb')as f:
                        features['text'][stage].append(pickle.load(f))
                    with open(audio_feat_path,'rb')as f:
                        features['audio'][stage].append(pickle.load(f))
                if not loaded:
                    loaded=True
                    with open(label_path,'rb')as f:
                        labels.append(pickle.load(f))

    for stage in stages:
        if fusion:
            features['fusion_cls'][stage]=torch.cat(features['fusion_cls'][stage],dim=0)
        else:
            features['text'][stage]=torch.cat(features['text'][stage],dim=0)
            features['audio'][stage]=torch.cat(features['audio'][stage],dim=0)
    labels=torch.cat(labels,dim=0)


    stage_alias={
        'raw':'Raw',
        'finetuned':'Fine-tuned',
    }
    for stage in stages:
        if fusion:
            plot_tsne(
                features['fusion_cls'][stage],
                labels,
                f't-SNE Visualization of {stage_alias[stage]} Early Fused Features',
                output_dir / f'{stage}_fusion_features.png'
            )
        else:
            # Visualize text features
            plot_tsne(
                features['text'][stage],
                labels,
                f't-SNE Visualization of {stage_alias[stage]} Text Features',
                output_dir / f'{stage}_text_features.png'
            )

            # Visualize audio features
            plot_tsne(
                features['audio'][stage],
                labels,
                f't-SNE Visualization of {stage_alias[stage]} Audio Features',
                output_dir / f'{stage}_audio_features.png'
            )