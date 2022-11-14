import data_pp_tools
import argparse
import pprint as pp
import cnn_lstm_model
import logging
import os
import time
import timeit
import torch
import tools
import train


parser = argparse.ArgumentParser(description='provide arguments')
parser.add_argument('--cuda_gpu', help='defulat GPU number', default=0)
parser.add_argument('--save_model', help='bool to indicate if to save NN model', default=1)
parser.add_argument('--log_en', help='bool to save log of the script in the working folder', default=1)
parser.add_argument('--plot_show', help='bool to indicate whether to show figures at the end of run', default=0)

# 添加数据解析和预处理的参数
parser.add_argument('--reshape_flag', help='If 1 then reshape data to 32x32 matrices', default=1)
parser.add_argument('--zero_pad', help='If true, zero pad train & validation sequences to max length, if Int than pad/subsample to Int', default=40)
parser.add_argument('--Nlabels', help='Number of used labels from the data set', default=11)

# CNN-LSTM模型参数
parser.add_argument('--Cin', help='Input channels', default=1)
parser.add_argument('--Cout1', help=' Channels Conv Layer 1', default=32)
parser.add_argument('--Cout2', help='Channels Conv layer 2', default=16)
parser.add_argument('--Cout3', help='Channels Conv layer 3', default=8)
parser.add_argument('--Lin_lstm', help='Input size to LSTM after FC', default=100)
parser.add_argument('--lstm_hidden_size', help='LSTM hidden size', default=100)
parser.add_argument('--lstm_num_layers', help='Number of LSTM layers', default=1)
parser.add_argument('--batch_size', help='Default batch size', default=64)

# 训练参数
parser.add_argument('--epochs', help='Num of epochs for training', default=40)
parser.add_argument('--adam_lr', help='Adam optimizer initial LR', default=1e-3)
parser.add_argument('--shuffle_data', help='Flag to shuffle data after each epoch', default=1)

args = vars(parser.parse_args())    # 解析添加的参数
pp.pprint(args)  # pp.pprint() 美观输出
train.main(args)   # 训练