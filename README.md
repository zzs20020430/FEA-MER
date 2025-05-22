# Multimodal Emotion Recognition with Contrastive Learning

This repository contains the implementation of a multimodal emotion recognition system that combines audio and text features using contrastive learning. The system uses a fusion of BERT and Wav2Vec2 models to achieve state-of-the-art performance on emotion recognition tasks.

## Overview

The project implements a novel approach to emotion recognition by:
- Using Wav2Vec2 for audio feature extraction
- Using RoBERTa for text feature extraction
- Implementing a multimodal fusion mechanism with BERT
- Applying contrastive learning to improve feature representation
- Incorporating binary classification for better modality alignment

## Requirements

- Conda
- CUDA-compatible GPU (recommended)

## Installation

1. Clone this repository:
```bash
git clone [repository-url]
cd [repository-name]
```

2. Create and activate the conda environment:
```bash
# Create environment from the provided yaml file
conda env create -f environment.yaml

# Activate the environment
conda activate base
```

## Data Preparation

The model is trained on the IEMOCAP dataset. You need to:
1. Download the IEMOCAP dataset
2. Place it in the appropriate directory structure
3. Update the configuration file in `config/dataset/iemocap.yaml`

## Model Architecture

The system consists of several key components:
- Audio Encoder: Wav2Vec2-based model for audio feature extraction
- Text Encoder: RoBERTa-based model for text feature extraction
- Multimodal Fusion: Custom BERT-based fusion module
- Projection Head: For contrastive learning
- Temperature Model: For contrastive loss calculation

## Training

To train the model:

1. Configure the parameters in `config/dataset/iemocap.yaml`
2. Run the training script:
```bash
python contrastive_learning_infornce_bert_fusion_bincls_emowav2vec.py
```

The training process includes:
- Contrastive learning loss
- Classification loss
- Binary classification loss
- Automatic model checkpointing

## Evaluation

The model is evaluated on the validation set after each epoch. Metrics include:
- Overall accuracy
- Weighted accuracy
- Per-class accuracy
- Validation loss

## Results

The model achieves state-of-the-art performance on the IEMOCAP dataset:
- Session 2: 85.10% accuracy
- Other sessions: ~80% accuracy

## Citation

If you use this code in your research, please cite:
```
[Citation details to be added]
```