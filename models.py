# encoding:utf-8

# 网络结构构建脚本
from torch import nn
from torchvision import models
import numpy as np
from transforms import *
from torch.nn.init import normal, constant
# from dataset import
import scipy.io
import csv
import pandas as pd
import os


def tensor_hook(grad):
    print('tensor hook')
    print('grad:', grad)

class myNet(nn.Module):  # 继承nn.Module
    def __init__(self, num_class, num_segments,
                 dropout=0.8,bi_flag=True
                 ):
        super(myNet, self).__init__()


        self.num_segments = num_segments  # 默认为3,视频分割成的段数
        self.reshape = True  # 是否reshape
        #self.before_softmax = before_softmax  # 模型是否在softmax前的意思？
        self.dropout = dropout  # dropout参数
        self.bi_num=2 if bi_flag else 1#if biLSTM is enabled
        print(("""
    Initializing myNet.
    myNet Configurations:
    num_segments:       {}
    dropout_ratio:      {}
        """.format(self.num_segments, self.dropout)))


    #structure of BN+BiLSTM+Temporal_Pooling
        self.stage1=nn.Sequential(
            #shape of input_var is [batchsize,channel,weight,height],aka [128,1,3,79872]
            nn.BatchNorm2d(1),

            #shape of input_var is still[128,1,3,79872]
            nn.MaxPool2d(3),
            #shape of input_var is[128,1,1,26624]
            nn.LeakyReLU(negative_slope=0.2)
        )

        self.lstmKernel=128
        self.lstmHier=2
        # shape of input_var is[1,128,26624]
        self.Bilstm=nn.LSTM(26624,self.lstmKernel,self.lstmHier,dropout=self.dropout,bidirectional=True)


        self.stage2=nn.Sequential(
            # shape of input_var is[128,1,1024]
            nn.BatchNorm2d(1),
            # shape of input_var is[128,1,1024]
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(self.dropout),
            nn.Linear(256,num_class),
            # shape of input_var is[128,1,5]
            nn.LeakyReLU(negative_slope=0.2),
            nn.Softmax()
            # shape of input_var is[128,1,5]
        )

        self.net=models.resnet101(pretrained=True)

    def forward(self, input):  # myNet类的forward函数定义了模型前向计算过程，也就是myNet的base_model+consensus结构

        #inference of BN+BiLSTM+Temporal_Pooling

        x1=self.stage1(input)

        x2=x1.squeeze(2)
        x3=x2.view(1,input.size()[0],26624)

        self.hidden = self.init_hidden(input.size()[0])

        #self.Bilstm.flatten_parameters() #cause prec@1 0.000,why?

        x4,self.hidden=self.Bilstm(x3,self.hidden)


        x5=x4.view(input.size()[0],1,256)

        x6=self.stage2(x5)
        #print(x6)
        x7=x6.squeeze()
        # shape of input_var is[128,5]

        return x7


    def init_hidden(self, batch_size):
        # 定义初始的hidden state:h0,c0
        return (torch.autograd.Variable(torch.zeros(self.lstmHier * self.bi_num, batch_size, self.lstmKernel).cuda()),
                torch.autograd.Variable(torch.zeros(self.lstmHier * self.bi_num, batch_size, self.lstmKernel).cuda()))



    def my_hook(self, module, grad_input, grad_output):
        print('doing my_hook')
        print('original grad:', grad_input)
        print('original outgrad:', grad_output)

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



        return [
            {'params': first_conv_weight, 'lr_mult':  1, 'decay_mult': 1,
             'name': "first_conv_weight"},
            {'params': first_conv_bias, 'lr_mult':  2, 'decay_mult': 0,
             'name': "first_conv_bias"},
            {'params': normal_weight, 'lr_mult': 1, 'decay_mult': 1,
             'name': "normal_weight"},
            {'params': normal_bias, 'lr_mult': 2, 'decay_mult': 0,
             'name': "normal_bias"},
            {'params': bn, 'lr_mult': 1, 'decay_mult': 0,
             'name': "BN scale/shift"},
        ]

    @property
    def crop_size(self):
       return self.input_size

    @property
    def scale_size(self):
        return self.input_size * 256 // 224

    def get_augmentation(self):
        return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75, .66]),
                                               GroupRandomHorizontalFlip(is_flow=False)])


class myCNN(nn.Module):  # 继承nn.Module
    def __init__(self, num_class,
                 dropout=0.8
                 ):
        super(myCNN, self).__init__()
        self.dropout = dropout  # dropout参数


    #structure of BN+BiLSTM+Temporal_Pooling
        self.stage1=nn.Sequential(
            #shape of input_var is [batchsize,channel,weight,height],aka [128,1,3,79872]
            nn.Conv2d(1,64,3,1,0),

            #shape of input_var is still[128,1,3,79872]
            nn.MaxPool2d(3),
            #shape of input_var is[128,1,1,26624]
            nn.ReLU(inplace=True)
        )


        # shape of input_var is[1,128,26624]
        self.Bilstm=nn.LSTM(26624,512,5,dropout=self.dropout,bidirectional=True)


        self.stage2=nn.Sequential(
            # shape of input_var is[128,1,1024]
            nn.BatchNorm2d(1),
            # shape of input_var is[128,1,1024]
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout),
            nn.Linear(1024,num_class),
            # shape of input_var is[128,1,5]
            nn.ReLU(inplace=True),
            nn.Softmax()
            # shape of input_var is[128,1,5]
        )

        self.net=models.resnet101(pretrained=True)

    def forward(self, input):  # myNet类的forward函数定义了模型前向计算过程，也就是myNet的base_model+consensus结构

        #inference of BN+BiLSTM+Temporal_Pooling

        x1=self.stage1(input)

        x2=x1.squeeze(2)
        x3=x2.view(1,input.size()[0],26624)

        self.hidden = self.init_hidden(input.size()[0])

        #self.Bilstm.flatten_parameters() #cause prec@1 0.000,why?

        x4,self.hidden=self.Bilstm(x3,self.hidden)


        x5=x4.view(input.size()[0],1,1024)

        x6=self.stage2(x5)

        x7=x6.squeeze()
        # shape of input_var is[128,5]

        return x7




