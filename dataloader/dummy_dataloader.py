import pandas
import math
import torch
from torch.utils.data import Dataset
import pandas as pd
import os
import numpy as np
from torch.utils.data import DataLoader
# if __name__=="__main__":
#     # from .tools import *
#     pass
# else:
#     from src.dataloaders.tools import *

from utils.tools import*


class DummyDataset(Dataset):
    # partition:train/test/val
    '''data={
        "text":[...],
        "audio":[...],
        "visual":[...],
        "personality":[...],
        "label":[...]
    }'''
    def __init__(self, args, num_samples=None):
        self.length=args.get('num_samples',num_samples) if num_samples is None else num_samples
        self.modalities=args.modalities
        self.feat_dims=args.feat_dims
        self.seqlen_interval={
            "text":(84,659),
            "audio":(946,6828),
            "visual":(36,273)
        }

    def generate(self,modal):
        seq_len_min, seq_len_max = self.seqlen_interval[modal]  # 序列长度范围
        feat_dim = self.feat_dims[modal]  # 特征维度

        # 随机选择一个序列长度
        seq_len = np.random.randint(seq_len_min, seq_len_max + 1)

        # 生成随机数据
        data = np.random.rand(seq_len, feat_dim)
        return data

    def __getitem__(self, idx):
        item= DotDict({
            attr:self.generate(attr) for attr in self.modalities
        })
        item.label=np.random.randint(0,2)
        return item

    def __len__(self):
        return self.length
    
    def _get_label_input(self):
        labels_embedding = np.arange(2)# [0,1]
        labels_mask = [1] * labels_embedding.shape[0]
        labels_mask = np.array(labels_mask)
        labels_embedding = torch.from_numpy(labels_embedding)
        labels_mask = torch.from_numpy(labels_mask)

        return labels_embedding, labels_mask

def pad_batch(batch):
    """
    特征填充，目前batch元素还是np.array，只有seqlen需要选出最大的以此为基准进行填充,feat_dim取一个元素计算就可以知道了。
    其中batch.shape == [batch_size, seq_len, feat_dim]
    
    :param batch: 一个批次的数据，每个元素是一个形状为 (seq_len, feat_dim) 的 np.array
    :return: 填充后的批次数据，形状为 (batch_size, max_seq_len, feat_dim)
    """
    # 获取批次大小
    batch_size = len(batch)
    
    # 计算特征维度
    feat_dim = batch[0].shape[-1]
    
    # 计算最大序列长度
    max_seq_len = max([sample.shape[0] for sample in batch])
    
    # 创建填充后的批次数据
    padded_batch = np.zeros((batch_size, max_seq_len, feat_dim), dtype=batch[0].dtype)
    
    # 填充数据
    for i, sample in enumerate(batch):
        seq_len = sample.shape[0]
        padded_batch[i, :seq_len, :] = sample
    
    return padded_batch

def pad_batch_with_masks(batch):
    """
    特征填充，目前batch元素还是np.array，只有seqlen需要选出最大的以此为基准进行填充,feat_dim取一个元素计算就可以知道了。
    其中batch.shape == [batch_size, seq_len, feat_dim]

    :param batch: 一个批次的数据，每个元素是一个形状为 (seq_len, feat_dim) 的 np.array
    :return: 填充后的批次数据，形状为 (batch_size, max_seq_len, feat_dim)，以及对应的掩码，形状为 (batch_size, max_seq_len)
    """
    # 获取批次大小
    batch_size = len(batch)

    # 计算特征维度
    feat_dim = batch[0].shape[-1]
    
    # 计算最大序列长度
    max_seq_len = max([sample.shape[0] for sample in batch])
    
    # 创建填充后的批次数据
    padded_batch = np.zeros((batch_size, max_seq_len, feat_dim), dtype=batch[0].dtype)
    
    # 创建掩码矩阵
    masks = np.zeros((batch_size, max_seq_len), dtype=np.int32)
    
    # 填充数据和掩码
    for i, sample in enumerate(batch):
        seq_len = sample.shape[0]
        padded_batch[i, :seq_len, :] = sample
        masks[i, :seq_len] = 1
    
    return padded_batch, masks

modalities=list(["text","audio","visual"])

def collate_fn(samples):
    # print("samples",samples[0])
    keys=set(samples[0].keys())
    batch={key:[] for key in keys}
    #原本的batch是list[dict]
    for sample in samples:
        for key in keys:
            batch[key].append(sample[key])
            # print(key,(batch[key]))

    collate_batch=DotDict({key:[] for key in keys})
    for k,v in batch.items():
        # if not isinstance(v,np.array):
        if k in modalities:   # 特征填充
            # collate_batch[k]=v=pad_batch(collate_batch[k])
            # print(f"pad_{k}: {v[0].shape}")
            collate_batch[k],collate_batch[f'{k}_mask']=pad_batch_with_masks(v)
        else:
            collate_batch[k]=np.array(v) if isinstance(v,list) else v

        collate_batch[k]=torch.FloatTensor(collate_batch[k]) if k in modalities+["label"] else collate_batch[k]
        # print(f"{k}_seq_len:{collate_batch[f'{k}'].shape}")
        if f'{k}_mask' in collate_batch.keys():
            collate_batch[f'{k}_mask']=torch.FloatTensor(collate_batch[f'{k}_mask'])
            
    # print(collate_batch)
    # return collate_batch.text,collate_batch.text_mask,collate_batch.visual,collate_batch.visual_mask,collate_batch.audio,collate_batch.audio_mask,collate_batch.label
    return collate_batch

# def get_dataloader(args,data:DotDict):
def get_dataloader(args):
    batch_size = args.batch_size
    train_dataset = DummyDataset(args,3000)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, num_workers=4, shuffle=True,collate_fn=collate_fn)

    test_dataset  = DummyDataset(args,1000)
    test_dataloader = DataLoader(dataset=test_dataset,batch_size=batch_size, num_workers=4, shuffle=False,collate_fn=collate_fn)

    return train_dataloader, test_dataloader


if __name__ =="__main__":
    # from mdpe_feature_file_loader_separate import*
    current_root=get_current_root(__file__)
    # config=read_yaml_to_dict(current_root.parent/"config/mdpe_sbert_hbb_vit_config.yaml")
    config=read_yaml_to_dict(current_root.parent/"config/mdpe_robertal_hbb_vit_config.yaml")
    # config=read_yaml_to_dict(current_root.parent/"config/mdpe_bai_hbb_vit_config.yaml")
    train_dataset=DummyDataset({},config)
    train_loader=DataLoader(train_dataset,batch_size=1,collate_fn=collate_fn)
    for batch in train_loader:
        pairs_text, pairs_mask, video, video_mask,audio, audio_mask, ground_label = batch
        print("label: ",ground_label.shape)
        print("text: ",pairs_text.shape,pairs_mask.shape)
        print("video: ",video.shape,video_mask.shape)
        print("audio: ",audio.shape,audio_mask.shape)
    pass