# encoding:utf-8

# 网络结构构建脚本
from torch import nn
import torchvision
import numpy as np
from transforms import *
from torch.nn.init import normal, constant
# from dataset import
import scipy.io
import csv
import pandas as pd
import os




class myNet(nn.Module):  # 继承nn.Module
    def __init__(self, num_class, num_segments,
                 base_model='resnet101', new_length=None,
                 consensus_type='avg', before_softmax=True,
                 dropout=0.8,
                 crop_num=1, partial_bn=True):
        super(myNet, self).__init__()

        self.num_segments = num_segments  # 默认为3,视频分割成的段数
        self.reshape = True  # 是否reshape
        self.before_softmax = before_softmax  # 模型是否在softmax前的意思？
        self.dropout = dropout  # dropout参数

        print(("""
    Initializing myNet with base model: {}.
    myNet Configurations:
    num_segments:       {}
    dropout_ratio:      {}
        """.format(base_model,  self.num_segments, self.new_length, consensus_type, self.dropout)))
        # 导入模型初始化


    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        :return:
        """
        # 重写train()来冻结BN参数
        super(myNet, self).train(mode)
        count = 0
        if self._enable_pbn:
            print("Freezing BatchNorm2D except the first one.")
            for m in self.base_model.modules():
                if isinstance(m, nn.BatchNorm2d):  # 如果base_model里出现一次BatchNorm2d,加一
                    count += 1
                    if count >= (2 if self._enable_pbn else 1):
                        m.eval()  # python中的eval()是将字符串转换成表达式计算，但不填参数是什么意思？

                        # shutdown update in frozen mode
                        m.weight.requires_grad = False
                        m.bias.requires_grad = False
                        # 梯度需求置否，使更新停止

    def partialBN(self, enable):
        self._enable_pbn = enable

    def get_optim_policies(self):  # 提取各层参数
        first_conv_weight = []
        first_conv_bias = []
        normal_weight = []
        normal_bias = []
        bn = []

        conv_cnt = 0
        bn_cnt = 0
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Conv1d):
                ps = list(m.parameters())  # 将参数转换成list保存进ps
                conv_cnt += 1  # 计数卷积层数量，出现2d或1d卷积就计一次
                if conv_cnt == 1:
                    first_conv_weight.append(ps[0])  # 提取第一个卷积层的weight
                    if len(ps) == 2:
                        first_conv_bias.append(ps[1])  # 如果参数是2，即bias不是0,取bias
                else:
                    normal_weight.append(ps[0])  # 取各层参数
                    if len(ps) == 2:
                        normal_bias.append(ps[1])
            elif isinstance(m, torch.nn.Linear):  # 同上，这是全连接层（线性变换层）
                ps = list(m.parameters())
                normal_weight.append(ps[0])
                if len(ps) == 2:
                    normal_bias.append(ps[1])

            elif isinstance(m, torch.nn.BatchNorm1d):
                bn.extend(list(m.parameters()))  # extend()用于在list末尾追加另一个序列的多个值，append()是添加单一对象
            elif isinstance(m, torch.nn.BatchNorm2d):
                bn_cnt += 1
                # later BN's are frozen
                if not self._enable_pbn or bn_cnt == 1:
                    bn.extend(list(m.parameters()))
            elif len(m._modules) == 0:
                if len(list(m.parameters())) > 0:
                    raise ValueError("New atomic module type: {}. Need to give it a learning policy".format(type(m)))

        # 保存参数

        # first_conv_weight_array=np.array(first_conv_weight)
        # print(first_conv_weight)
        # np.savetxt('./first_conv_weight.txt',fmt=['%s']*first_conv_weight_array.shape[1],newline='\n')
        # print('Finishing saving txt file')

        return [
            {'params': first_conv_weight, 'lr_mult': 5 if self.modality == 'Flow' else 1, 'decay_mult': 1,
             'name': "first_conv_weight"},
            {'params': first_conv_bias, 'lr_mult': 10 if self.modality == 'Flow' else 2, 'decay_mult': 0,
             'name': "first_conv_bias"},
            {'params': normal_weight, 'lr_mult': 1, 'decay_mult': 1,
             'name': "normal_weight"},
            {'params': normal_bias, 'lr_mult': 2, 'decay_mult': 0,
             'name': "normal_bias"},
            {'params': bn, 'lr_mult': 1, 'decay_mult': 0,
             'name': "BN scale/shift"},
        ]


    def forward(self, input):  # myNet类的forward函数定义了模型前向计算过程，也就是myNet的base_model+consensus结构





    @property
    def crop_size(self):
        return self.input_size

    @property
    def scale_size(self):
        return self.input_size * 256 // 224

    def get_augmentation(self):
     #   if self.modality == 'RGB':
        return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75, .66]),
                                                   GroupRandomHorizontalFlip(is_flow=False)])
     #   elif self.modality == 'Flow':
     #       return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75]),
     #                                              GroupRandomHorizontalFlip(is_flow=True)])
     #   elif self.modality == 'RGBDiff':
     #       return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75]),
     #                                              GroupRandomHorizontalFlip(is_flow=False)])