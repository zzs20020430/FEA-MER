if __name__!="__main__":
    from utils.tools import *
else:#for test
    from tools import *

import torch
import torch.nn as nn
import torchaudio
import numpy as np
import os
import pickle
from torch.utils.data import DataLoader, Dataset

label_list=['neutral', 'anger', 'joy', 'surprise', 'sadness', 'disgust', 'fear']
label_dict={label:idx for idx,label in enumerate(label_list)}
protocols=["train", "dev", "test"]
def load_cache(data:dict, cache_path):
    # print(data)
    # merge cache
    if os.path.exists(cache_path):
        try:
            cache = pickle.load(open(cache_path, 'rb'))
            data.update(cache)
            return True
        except Exception as e:
            print(f"Failed to load cache from {cache_path}: {e}")
            return False
    else:
        print(f"Cache file {cache_path} does not exist.")
        return False
    
def save_cache(data:dict,cache_path):
    pickle.dump(data,open(cache_path,'wb'))

def load_protocol(data_dir:Path, annotation_path):
    data_list=read_data_sheet(annotation_path)
    samples=[]
    sample_num=len(data_list)
    for index in range(sample_num):
        dia_id = data_list.at[index, 'Dialogue_ID']
        utt_id = data_list.at[index, 'Utterance_ID']
        text = data_list.at[index, 'Utterance']
        filename = f'dia{dia_id}_utt{utt_id}.wav'
        waveform, sample_rate =torchaudio.load(str(data_dir/filename))
        duration=waveform.shape[1]/sample_rate
        samples.append({
            'text':text,
            'audio':waveform.mean(dim=0).numpy(),                               #单声道处理
            'filename':filename,
            'dia_id':dia_id,
            'utt_id':utt_id,
            'duration':duration,
            'emotion':data_list.at[index,'Emotion'],
            'sentiment':data_list.at[index,'Sentiment'],
            'sample_rate':sample_rate
        })
    return samples

def load_aug_data(args):
    dataset_path=Path(args.dataset_path)
    cache_path=dataset_path/"data_cache/aug_data.pkl" 
    aug_data={}
    if load_cache(aug_data, cache_path):return aug_data
    annotation_dir=dataset_path/"annotation"
    for protocol in protocols[:-1]:
        data_dir,annotation_path=dataset_path/f"{protocol}_aug", annotation_dir/f"{protocol}_sent_emo.tsv"
        data_list=read_data_sheet(annotation_path)
        aug_data[protocol]={}
        sample_num=len(data_list)
        for index in range(sample_num):
            dia_id = data_list.at[index, 'Dialogue_ID']
            utt_id = data_list.at[index, 'Utterance_ID']
            text = data_list.at[index, 'Aug_Utterance']
            filename = f'dia{dia_id}_utt{utt_id}.wav'
            waveform, sample_rate =torchaudio.load(str(data_dir/filename))
            aug_data[protocol][filename]={
                'text':text,
                'audio':waveform.mean(dim=0).numpy(),                               #单声道处理
            }
    # print(aug_data)
    save_cache(aug_data,cache_path)
    return aug_data

def load_raw_data(args,aug=False):
    dataset_path=Path(args.dataset_path)
    data_path=dataset_path/"data_cache/raw_data.pkl" 
    raw_data={}
    loaded= load_cache(raw_data,data_path)
    
    annotation_dir=dataset_path/"annotation"
    if not loaded:
        for protocol in protocols:
            raw_data[protocol]=load_protocol(dataset_path/f"{protocol}", annotation_dir/f"{protocol}_sent_emo.tsv")

        save_cache(raw_data,data_path)

    if args.get('augment',False):
        aug_data=load_aug_data(args)
        for protocol in protocols[:-1]:
            for sample in raw_data[protocol]:
                filename=sample['filename']
                if filename in aug_data[protocol].keys():
                    sample['text_aug']=aug_data[protocol][filename]['text']
                    sample['audio_aug']=aug_data[protocol][filename]['audio']

    return raw_data

def data_filter(data:list,args):
    [item.update({'label':label_dict[item['emotion']]}) for item in data]
        # 修改过滤逻辑，将超过max_duration的音频截断
    filtered_data = []
    for item in data:
        if item['label'] < args.num_classes:
            #超长样本
            if item['duration'] > args.max_duration:
                if args.truncation:
                    # 计算需要截断的采样点数
                    # max_samples = int(args.max_duration * 16000)
                    max_samples = int(args.max_duration * item['sample_rate'])
                    # 截断音频
                    item['audio'] = item['audio'][:max_samples]
                    if item.get('text_aug',None) is not None:
                        item['audio_aug'] = item['audio_aug'][:max_samples]
                    # 更新duration
                    item['duration'] = args.max_duration
                else:# 不截断则直接跳过加载
                    continue
            filtered_data.append(item)
    return filtered_data


class MELDDataset(Dataset):
    def __init__(self, data_list, args):
        self.num_classes=args.num_classes

        # 直接加载保存的pkl文件中的数据
        # self.data=[item for item in data_list if item['label'] in label_set_mapper[args.num_classes]] #使用raw label过滤
        # self.data=[item for item in data_list if (item['label'] in label_set_mapper[args.num_classes]) and (item['duration']<=args.max_duration)] #使用raw label过滤
        self.data=data_filter(data_list,args)
        self.length=len(self.data)
    def __len__(self):
        return self.length
    def __getitem__(self, index):
        # self.data[index]['label']=label_mapper[self.num_classes](self.data[index]['label'])
        return self.data[index]
    
    def statistics(self):
        # print('-'*40)
        for label in range(self.num_classes):
            print(f"{label_list[label]}: {len([_ for _ in filter(lambda x:x['label']==label,self.data)])}/{self.length}")
        print('')
        # print('-'*40)
        
    def aug_statistics(self):
        return any([item.get('audio_aug',None) is not None for item in self.data])
    
    def get_label_weights(self):
        if getattr(self,'label_weights',None) is None:
            self.label_weights={}
            for label in range(self.num_classes):
                self.label_weights[label]=len([_ for _ in filter(lambda x:x['label']==label,self.data)])/self.length
        return self.label_weights

def collate_fn(batch):
    keys=batch[0].keys()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch={key:[sample[key] for sample in batch]for key in keys}

    # sampling_rate=batch["sample_rate"][0]

    #也许在外部提取以启用num_workers
    # batch["audio"],batch["audio_cls"]=audio_encoder(batch["audio"])
    # batch["text"],batch["text_cls"]=text_encoder(batch["text"])

    batch['label'] = torch.tensor(batch['label'], dtype=torch.long).to(device)
    # 获取音频维度特征 (audio dimension)，并将其移动到 GPU
    # raw_audio_dimensions = torch.stack([torch.tensor(sample['raw_sentimental_density']) for sample in batch]).to(device)
    # aug_audio_dimensions = torch.stack([torch.tensor(sample['aug_sentimental_density']) for sample in batch]).to(device)

    return DotDict(batch)

def get_dataloader(args, data=None):
    if data is None:
        data = load_raw_data(args)
    # print(data['session1'][0].keys())
    train_data=data["train"]
    test_data=data["test"]
    if args.train_dev:
        train_data+=data["dev"]
    elif args.test_dev:
        test_data+=data["dev"]

    batch_size = args.batch_size
    train_dataset = MELDDataset(train_data,args)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size,
                                #   num_workers=4,
                                    shuffle=True,collate_fn=collate_fn,drop_last=True)

    test_dataset  = MELDDataset(test_data,args)
    test_dataloader = DataLoader(dataset=test_dataset,batch_size=batch_size,
                                #  num_workers=4,
                                   shuffle=False,collate_fn=collate_fn,drop_last=True)
    trainset_len=len(train_dataset)
    testset_len=len(test_dataset)
    print(f"num_classes: {args.num_classes}")
    print(f"training {'with dev set'if args.train_dev else ', will test on test+dev set' if args.test_dev else ''}")
    print(f"train: {trainset_len}/{trainset_len+testset_len}")
    print(f"test: {testset_len}/{trainset_len+testset_len}")

    # print('train_set.statistics:')
    # train_dataset.statistics()
    # print('test_set.statistics:')
    # test_dataset.statistics()
    # print(train_dataset.aug_statistics())


    return train_dataloader, test_dataloader

if __name__ =="__main__":
    args=DotDict({
        'dataset_path':'/18t/data/home/zzs/proj/SpeechEmotion/data/meld',
        'test_session':1,
        'batch_size':4,
        'num_classes':4,
        'max_duration':25,   #second
        'train_dev':True,
        'test_dev':False,
        'augment':True
    })
    # load_raw_data(args)
    train_loader,val_loader=get_dataloader(args)
    for batch in val_loader:
        # print(batch)
        pass