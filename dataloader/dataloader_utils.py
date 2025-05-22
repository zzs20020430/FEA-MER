import numpy as np
import torch
from utils.tools import *

modalities=set(["text","audio","visual"])
no_padding_attrs=set(["filename"])
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

def collate_fn(samples):
    '''
    samples: list[dict]->dict[str,any]
    '''
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
            collate_batch[k]=np.array(v) if isinstance(v,list) and k not in no_padding_attrs else v
        collate_batch[k]=torch.FloatTensor(collate_batch[k])
        # print(f"{k}_seq_len:{collate_batch[f'{k}'].shape}")
        if f'{k}_mask' in collate_batch.keys():
            collate_batch[f'{k}_mask']=torch.FloatTensor(collate_batch[f'{k}_mask'])
            
    # print(collate_batch)
    # return collate_batch.text,collate_batch.text_mask,collate_batch.visual,collate_batch.visual_mask,collate_batch.audio,collate_batch.audio_mask,collate_batch.label
    return collate_batch