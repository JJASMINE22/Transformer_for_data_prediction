# -*- coding: UTF-8 -*-
'''
@Project ：transformer_for_data_prediction
@File    ：predict.py
@IDE     ：PyCharm 
@Author  ：XinYi Huang
'''

import torch
import numpy as np
import config as cfg
import matplotlib.pyplot as plt
from torch import nn
from Transformer import TransFormer
from utils.data_generator import Generator


if __name__ == '__main__':

    transformer = TransFormer(target_size=cfg.target_size,
                              source_size=cfg.source_size,
                              embedding_size=cfg.embedding_size,
                              multihead_num=cfg.multihead_num,
                              num_layers=cfg.num_layers,
                              drop_rate=cfg.drop_rate,
                              weight_decay=cfg.weight_decay,
                              learning_rate=cfg.learning_rate,
                              load_ckpt=cfg.load_ckpt,
                              ckpt_path=cfg.checkpoint_path + "\\模型文件",
                              device=cfg.device)

    data_gen = Generator(txt_path=cfg.text_path,
                         batch_size=cfg.batch_size,
                         ratio=cfg.ratio,
                         time_seq=cfg.time_seq)

    _, source, target = data_gen.preprocess()

    src = torch.tensor(source[-1][np.newaxis, ...]).float().to(cfg.device)
    tgt = None

    # get predictions based on roll times
    for i in range(cfg.roll_time):
        tgt = transformer.model(src, tgt)

    tgt = tgt.squeeze(0).cpu().detach().numpy()
    for j in range(tgt.shape[-1]):
        plt.subplot(9, 1, j+1)
        plt.plot(tgt[:, j], color='r',  marker='*', linewidth=1,
                 label='feature_{}_prediction'.format(j+1))
        plt.grid()
        plt.legend(loc='upper right', fontsize='x-small')
    plt.show()
