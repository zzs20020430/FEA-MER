import torch
import torch.nn as nn
import torchaudio
import numpy as np
import os
import pickle
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
if __name__!="__main__":
    from utils.tools import *
else:#for test
    from tools import *

#ang:0
#hap:1
#sad:2
#neu:3

# labels=['neu','ang','hap','sad','exc','fru','fea','sur','xxx']
# labels=['neu','hap','sad','ang','exc','fru','fea','sur','xxx']
label_list=['ang','hap','sad','neu','exc','fru','fea','sur','xxx']
label_set=set(label_list)
label_dict={label:idx for idx,label in enumerate(label_list)}

#merge class
def merge_class():
    label_dict['exc']=label_dict['hap'] #exec->happ
    label_dict['oth']=label_dict['xxx'] #other->xxx
    label_dict['dis']=label_dict['xxx'] #disgust->xxx

def read_annotation(file_path):
    blocks = []
    current_block = []

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            if line.startswith('%'):
                continue
            line = line.strip()
            if line:
                current_block.append(line)
            else:
                if current_block:
                    blocks.append(current_block[0])
                    current_block = []
    
    # Add the last block if there's no trailing newline
    if current_block:
        blocks.append(current_block[0])
    # Extract the first line of each block
    # print(blocks)
    # print(len(blocks))

    extracted_lines = {block.split('\t')[1]:block.split('\t')[2] for block in blocks} #{filename:label}
    return extracted_lines

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

#load non-augmented raw data
def load_raw_data(args):
    dataset_path=Path(args.dataset_path)
    data_path=dataset_path/"data_cache/raw_data.pkl" 
    raw_data={}
    loaded= load_cache(raw_data,data_path)

    #session[1~5]
    for session_idx in range(1,6 if not loaded else 1):
        raw_data[f'session{session_idx}']=[]
        session_path=dataset_path/f"Session{session_idx}"
        transcription_dir=session_path/f"dialog/transcriptions"
        sentence_dir=session_path/f"sentences/wav"
        label_dir=session_path/"dialog/EmoEvaluation"

        label_list=sorted([label_path for label_path in label_dir.iterdir() if label_path.is_file()],key=lambda x:x.stem)                    #label
        transcript_list=sorted(list(transcription_dir.iterdir()),key=lambda x:x.stem)       #text
        sentence_list=sorted(list(sentence_dir.iterdir()),key=lambda x:x.stem)              #audio

        for transcript,sentences,sentence_label in zip(transcript_list,sentence_list,label_list):
            dia_transcript={ line.split(': ')[0].split(' ')[0]:line.split(': ')[-1] for line in read_text_list(transcript)}#{filename:text}
            sentence_annotation=read_annotation(sentence_label)
            sentence_paths=sorted([sentence for sentence in sentences.iterdir() if sentence.suffix=='.wav'],key=lambda x:x.stem)
            # print(sentence_paths)
            #遍历音频文件，按照对话句子进一步划分，所以还需要遍历子文件夹
            for sentence in sentence_paths:
                waveform, sample_rate =torchaudio.load(sentence)
                duration=waveform.shape[1]/sample_rate

                raw_data[f'session{session_idx}'].append({
                    'text':dia_transcript[sentence.stem],            #text
                    # 'waveform':waveform.numpy().squeeze(0),          #audio
                    'audio':waveform.numpy().squeeze(0),            #audio
                    'label':sentence_annotation[sentence.stem],      #label
                    'duration':duration,                             #seconds
                    'filename':sentence.stem,
                    'sample_rate':sample_rate
                })
                # print(raw_data[f'session{session_idx}'])  
    if not loaded:
        save_cache(raw_data,data_path)

    if args.get('augment',False):
        data_filenames=set([data['filename']for session,data_list in raw_data.items() for data in data_list])
        aug_data=load_aug_data(data_filenames,dataset_path)
        for session,sample_list in raw_data.items():
            if f"session{args.test_session}"==session:#不增强测试集
                continue
            for sample in sample_list:
                filename=sample['filename']
                if filename in aug_data[session].keys():
                    aug_sample=aug_data[session][filename]
                    sample['text_aug']=aug_sample['text']
                    sample['audio_aug']=aug_sample['audio'][random.randint(0,len(aug_sample['audio'])-1)]
                else:
                    continue
    return raw_data

#一个音频有多种增强
def load_aug_data(data_filenames,dataset_path:Path):
    cache_path=dataset_path/"data_cache/aug_data.pkl" 
    data_dir=dataset_path/"iemocap_aug/out"
    aug_data={f"session{idx}":{} for idx in range(1,6)}

    if load_cache(aug_data,cache_path):
        return aug_data
    
    session_mapper={f'Ses0{idx}':idx for idx in range(1,6)}
    data_annotation=read_data_sheet(dataset_path/"iemocap.csv")
    data_annotation.set_index('FileName', inplace=True)

    for aug_file in data_dir.iterdir():
        aug_filename=aug_file.stem          # {aug_prefix}_{original_filename}.wav
        raw_filename=f"Ses{aug_filename.split('Ses')[-1]}"
        if raw_filename not in data_filenames:
            print(f"{raw_filename} does not exsist in raw_data, skip...")
            continue
        session_prefix=raw_filename[:5]
        session_idx=session_mapper[session_prefix]
        row = data_annotation.loc[raw_filename]
        waveform,_=torchaudio.load(aug_file)
        if aug_data[f'session{session_idx}'].get(raw_filename,None) is None:
            aug_data[f'session{session_idx}'][raw_filename]={
                'text':row.AugmentedText,
                'audio':[waveform.numpy().squeeze(0)]
            }
        else:
            aug_data[f'session{session_idx}'][raw_filename]['audio'].append(waveform.numpy().squeeze(0))
    save_cache(aug_data,cache_path)
    return aug_data

def data_filter(data:list,args):
    if args.num_classes==4:
        merge_class()
    [item.update({'label':label_dict[item['label']]}) for item in data]
        # 修改过滤逻辑，将超过max_duration的音频截断
    filtered_data = []
    for item in data:
        if item['label'] < args.num_classes:
            #超长样本
            if item['duration'] > args.max_duration:
                if args.truncation:
                    # 计算需要截断的采样点数
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


class IEMOCAPDataset(Dataset):
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
    
    def get_label_weights(self):
        if self.label_weights is None:
            self.label_weights={}
            for label in range(self.num_classes):
                self.label_weights[label]=len([_ for _ in filter(lambda x:x['label']==label,self.data)/{self.length}])
        return self.label_weights
    
    def statistics(self):
        for label in range(self.num_classes):
            print(f"{label_list[label]}: {len([_ for _ in filter(lambda x:x['label']==label,self.data)])}/{self.length}")
        print('')

    
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
    test_session=args.test_session if isinstance(args.test_session,str)else f"session{args.test_session}"
    if data is None:
        data = load_raw_data(args)
    # print(data['session1'][0].keys())

    test_data=data[test_session]
    train_data=sum([session for key,session in data.items() if key!=test_session],[])
    batch_size = args.batch_size
    train_dataset = IEMOCAPDataset(train_data,args)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size,
                                #   num_workers=4,
                                    shuffle=True,collate_fn=collate_fn,drop_last=True)

    test_dataset  = IEMOCAPDataset(test_data,args)
    test_dataloader = DataLoader(dataset=test_dataset,batch_size=batch_size,
                                #  num_workers=4,
                                   shuffle=False,collate_fn=collate_fn,drop_last=True)
    # trainset_len=len(train_dataset)
    # testset_len=len(test_dataset)
    # print(f"training on {test_session}")
    # print(f"train:{trainset_len}/{trainset_len+testset_len}")
    # print(f"test:{testset_len}/{trainset_len+testset_len}")
    return train_dataloader, test_dataloader

def get_label_weights(dataloader:DataLoader):
    dataset=dataloader.dataset
    return dataset.get_label_weights()

if __name__=="__main__":

    args=DotDict({
        'dataset_path':'/18t/data/home/zzs/proj/SpeechEmotion/data/IEMOCAP语料库',
        'test_session':1,
        'batch_size':4,
        'num_classes':4,
        'max_duration':25,   #second
        'augment':True
    })
    # print(int("session1"))
    # load_raw_data(args)
    # print(load_raw_data(args))
    # print(sum([[1],[2],[3]],[]))


    # from model.network.audio_encoder import AudioEncoder
    # from model.network.text_encoder import TextEncoder

    # model_dir=Path("/18t/data/home/wangchai/SpeechEmotion/models/")
    # audio_model_path =model_dir/ 'wav2vec2-large-uncased'
    # text_model_path=model_dir/'roberta-large-uncased'
    # device=get_device()
    # text_encoder = TextEncoder(model_path=text_model_path).to(device)

    train_loader,val_loader=get_dataloader(args)
    for batch in val_loader:
        # print(batch)
        pass