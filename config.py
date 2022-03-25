# -*- coding: UTF-8 -*-
'''
@Project ：transformer_for_data_prediction
@File    ：config.py
@IDE     ：PyCharm 
@Author  ：XinYi Huang
'''
import torch

# ===data loader===
text_path='数据路径'
epoches=200
batch_size=8
ratio=0.7
time_seq=7

# ===model===
target_size=9
source_size=256
embedding_size=256
multihead_num=4
num_layers=2
drop_rate=0.4
learning_rate = 3e-4
weight_decay = 5e-4
per_sample_interval = 50
device = torch.device('cuda') if torch.cuda.is_available() else None
checkpoint_path = '模型文件路径'
sample_path = '预测效果路径'
load_ckpt = True

# ===prediction===
roll_time = 10
