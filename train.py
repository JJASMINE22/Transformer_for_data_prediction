# -*- coding: UTF-8 -*-
'''
@Project ：transformer_for_data_prediction
@File    ：train.py
@IDE     ：PyCharm 
@Author  ：XinYi Huang
'''
import torch
import config as cfg
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

    train_func = data_gen.generate(training=True)
    validate_func = data_gen.generate(training=False)

    for epoch in range(cfg.epoches):

        for i in range(data_gen.get_train_len()):
            sources, targets = next(train_func)
            logits = targets[:, :-1]

            transformer.train(sources, logits, targets)
            if not i % cfg.per_sample_interval and i:
                transformer.generate_sample(sources, logits, targets, i)

        print('Epoch{:0>3d} train loss is {:.5f}'.format(epoch+1, transformer.train_loss / data_gen.get_train_len()))

        torch.save({'state_dict': transformer.model.state_dict(),
                    'loss': transformer.train_loss / data_gen.get_train_len()},
                   cfg.checkpoint_path + '\\Epoch{:0>3d}_train_loss{:.5f}.pth.tar'.format(
                       epoch + 1, transformer.train_loss / data_gen.get_train_len()
                   ))
        transformer.train_loss = 0

        for i in range(data_gen.get_val_len()):
            sources, targets = next(validate_func)
            logits = targets[:, :-1]

            transformer.validate(sources, logits, targets)
        print('Epoch{:0>3d} validate loss is {:.5f}'.format(epoch+1, transformer.val_loss / data_gen.get_val_len()))

        transformer.val_loss = 0
