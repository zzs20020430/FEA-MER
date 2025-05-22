import torch
import torch.nn as nn
import torchaudio
from transformers import RobertaTokenizer, RobertaModel, Wav2Vec2Processor
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2Model, Wav2Vec2PreTrainedModel
from pathlib import Path
import numpy as np

if __name__=="__main__":
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] ="1"

# class RegressionHead(nn.Module):
#     r"""Classification head."""
#     def __init__(self, config):
#         super().__init__()
#         self.dense = nn.Linear(config.hidden_size, config.hidden_size)
#         self.dropout = nn.Dropout(config.final_dropout)
#         self.out_proj = nn.Linear(config.hidden_size, config.num_labels)
#     def forward(self, features, **kwargs):
#         x = features
#         x = self.dropout(x)
#         x = self.dense(x)
#         x = torch.tanh(x)
#         x = self.dropout(x)
#         x = self.out_proj(x)
#         return x


# Get text embeddings
class TextEncoder(nn.Module):
    def __init__(self, tokenizer, text_model):
        super(TextEncoder, self).__init__()
        self.tokenizer = tokenizer
        self.text_model = text_model

    def forward(self, text_inputs):
        text_embeddings = self.tokenizer(text_inputs, return_tensors='pt', padding=True, truncation=True,
                                         max_length=80).to(device)
        text_output = self.text_model(**text_embeddings)
        text_seq_embedding, text_cls_embeddings = text_output[0], text_output[1]
        return text_seq_embedding, text_cls_embeddings

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_dir=Path("/18t/data/home/wangchai/SpeechEmotion/models/")
audio_model_path =model_dir/ 'wav2vec2-large-uncased'

### 使用processor来获取语音的tokenizer
def audio_tokenizer(x: np.ndarray, sampling_rate: int) -> np.ndarray:
    inputs = processor(x, sampling_rate=sampling_rate,return_attention_mask=True, padding=True,return_tensors='pt')
    input_values = inputs['input_values']#[0]  # 输入的音频特征 (input_ids)
    attention_mask = inputs['attention_mask']#[0]  # attention mask
    return input_values,attention_mask

###  加载预训练的Wav2vec2模型，输出的hidden_states1是最后一层隐藏层的沿着时间序列方向平均池化后的结果结果，logits是池化向量映射到三维连续情感空间的结果。
###  如果想要取出，hidden_states的非池化结果，即维度为[batch, seq_len, 1024],只需要取出 hidden_states0 = outputs[0]
class AudioModel(Wav2Vec2PreTrainedModel):
    r"""Speech emotion classifier."""
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.wav2vec2 = Wav2Vec2Model(config)
        # self.classifier = RegressionHead(config)
        self.init_weights()
    def forward(self,input_values,attention_mask=None):
        outputs = self.wav2vec2(input_values,attention_mask=attention_mask)
        hidden_states0 = outputs[0]
        hidden_states1 = torch.mean(hidden_states0, dim=1)
        # logits = self.classifier(hidden_states1)
        return hidden_states0, hidden_states1#, logits
    
class AudioEncoder(nn.Module):
    r"""Speech emotion classifier."""
    def __init__(self, processor,audio_model,sample_rate=16000):
        super(AudioEncoder,self).__init__()
        self.processor = processor
        self.audio_model = audio_model
        self.sample_rate=sample_rate

    def forward(self,audio_inputs):
        # inputs = self.processor(audio_inputs, sampling_rate=self.sample_rate,return_attention_mask=True, padding=True)#, return_tensors='pt')
        inputs = self.processor(audio_inputs, sampling_rate=self.sample_rate,return_attention_mask=True, padding=True, return_tensors='pt')
        input_values = inputs['input_values']#[0]  # 输入的音频特征 (input_ids)
        attention_mask = inputs['attention_mask']#[0]  # attention mask
        # print(input_values)
        # print(attention_mask)
        return self.audio_model(input_values.cuda(),attention_mask.cuda())
    
# feat_encoder tensor([[ 0.0585,  0.0747,  0.0838,  ..., -0.0214,  0.0262,  0.1552],
    # [-0.0121, -0.0178, -0.0240,  ...,  0.0000,  0.0000,  0.0000]])
    
# feat_encoder [array([ 0.05848019,  0.07474868,  0.08375445, ..., -0.02140973,
#         0.02623371,  0.15521963], dtype=float32), array([-0.01207236, -0.01782476, -0.02401965, ...,  0.        ,
#         0.        ,  0.        ], dtype=float32)]

processor = Wav2Vec2Processor.from_pretrained(audio_model_path)
audio_model = AudioModel.from_pretrained(audio_model_path).to(device)
audio_encoder=AudioEncoder(audio_model=audio_model,processor=processor).to(device)

# RoBERTa configuration
text_model_path=model_dir/'roberta-large-uncased'
roberta_tokenizer = RobertaTokenizer.from_pretrained(text_model_path)
roberta_model = RobertaModel.from_pretrained(text_model_path).to(device)
text_encoder=TextEncoder(roberta_tokenizer,roberta_model).to(device)

if __name__=="__main__":

    # text="Here we must have died alone, a long long time ago."
    texts=["Here we must have died alone, a long long time ago.","Oh no, not me. I never lost control.","You are face, to face, with the man who sold the world."]
    print('-'*10,"text batch","-"*10)
    out_seqs,out_clses=text_encoder(texts)
    print('out_seq',out_seqs,out_seqs.shape)      #padding none zero
    print('out_cls',out_clses,out_clses.shape)
    # print(torch.max(out_seq,dim=1)[0].shape)
    # print("----------------separate text------------------")
    # for idx,text in enumerate(texts):
    #     out_seq,out_cls=text_encoder(text)
    #     print('out_seq',out_seq,out_seq.shape)
    #     print('out_cls',out_cls,out_cls.shape)
    #     print(torch.max(out_seq,dim=1)[0].shape)
    #     for jdx in range(out_seq.shape[1]):
    #         print((out_seq[0][jdx]==out_seq[idx][jdx]).tolist())

    sentences=["/home/wangchai/SpeechEmotion/data/IEMOCAP语料库/Session1/sentences/wav/Ses01F_impro01/Ses01F_impro01_F006.wav","/home/wangchai/SpeechEmotion/data/IEMOCAP语料库/Session1/sentences/wav/Ses01F_impro01/Ses01F_impro01_F005.wav"]
    waveforms=[torchaudio.load(sentence)[0].squeeze(0).numpy() for sentence in sentences]

    _, sample_rate =torchaudio.load(sentences[0])
    print('-'*10,"audio batch","-"*10)
    # print(waveforms,[wave.shape for wave in waveforms])
    # audio_inputs=processor(waveforms,
    #                        sampling_rate=sample_rate,
    #                        return_attention_mask=True,
    #                        padding=True,
    #                     #    max_length=max([wave.shape[0] for wave in waveforms]),
    #                     #    padding='max_length'        #transformers 4.40:[enum], 4.37:bool
    #                        )
    # input_values=audio_inputs['input_values']
    # attention_masks=audio_inputs['attention_mask']
    # print('input_values\n',input_values)            #padding zero
    # print('attention_mask\n',attention_masks)
    # print(audio_model(input_values,attention_masks))
    print(audio_encoder(waveforms))

    # print('-'*10,"separate audio","-"*10)
    # for idx,wave in enumerate(waveforms):
    #     audio_input=processor(wave, sampling_rate=sample_rate,return_attention_mask=True, padding=True)
    #     # input_value=audio_input['input_values'][0]
    #     # attention_mask=audio_input['attention_mask'][0]
    #     input_value,attention_mask=audio_tokenizer(wave,sample_rate)
    #     print('input_values\n',input_value)
    #     print('attention_mask\n',attention_mask)
    #     for jdx in range(input_value.shape[0]):
    #         print((input_value[jdx]==input_values[idx][jdx]).tolist())


