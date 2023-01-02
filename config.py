import torch
import argparse
import os
from utils import set_logger

d_model = 512
n_heads = 8
n_layers = 6
d_k = 64
d_v = 64
d_ff = 2048
dropout = 0.1
padding_idx = 0
bos_idx = 2
eos_idx = 3
src_vocab_size = 32000
tgt_vocab_size = 32000
batch_size = 16
epoch_num = 40
early_stop = 5
lr = 3e-4

type = "test"

# greed decode的最大句子长度
max_len = 60
# beam size for bleu
beam_size = 3
# Label Smoothing
use_smoothing = False
# NoamOpt
use_noamopt = True

data_dir = './data'
train_data_path = './data/json/train.json'
dev_data_path = './data/json/dev.json'
test_data_path = './data/json/test.json'
model_path = './log/model.pth'
log_path = './log/train.log'
output_path = './log/output.txt'

logger = set_logger(log_path)
# gpu_id and device id is the relative id
# thus, if you wanna use os.environ['CUDA_VISIBLE_DEVICES'] = '2, 3'
# you should set CUDA_VISIBLE_DEVICES = 2 as main -> gpu_id = '0', device_id = [0, 1]


# parser = argparse.ArgumentParser()
# parser.add_argument('--gpu', default=4, type=int, help=" ")
# args = parser.parse_args()


# 限制代码能看到的GPU个数，GPU的实际id，字符串形式，device_count=2
os.environ['CUDA_VISIBLE_DEVICES'] = '3, 4, 5, 6'  # 顺序很重要
# device_ids = [0, 1]，GPU程序内id，0对应5号卡，1对应6号卡，
device_ids = list(range(torch.cuda.device_count()))
# model和data都由master GPU(0对应的5号卡) 分发，负载不均衡，效率低


# 指定GPU，顺序很重要，影响哪个（实际）卡号GPU作为（逻辑）master GPU
# 命令行中：CUDA_VISIBLE_DEVICES = 0, 2, 3  python main.py
# 程序中，os.environ['CUDA_VISIBLE_DEVICES'] = "0, 2, 3"
# IPython 或 jupyter notebook 中，%env CUDA_VISIBLE_DEVICES= 0, 2, 3


# data, def function, model迁移到GPU上, 尤其是多GPU下DP方式
"""
1. .cuda 和 .cpu()
1.1 .cpu data.cpu(), func.cpu(), model.cpu()
1.2 .cuda()
1.2.1 单个GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'  
    
    data.cuda(), func.cuda(), model.cuda()
1.2.1 多个GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = '2, 3, 4, 5'
    device_ids = list(range(torch.cuda.device_count()))  # [0, 1, 2, 3]
    model = torch.nn.DataParallel(model, device_ids=device_ids)
    
    model.cuda(), func.cuda(), data.cuda()  # 迁移到default中，即第一块（逻辑0）GPU中，实际卡号2GPU中
    .cuda() 迁移到default GPU中
    .cuda(1) 迁移到第二块 GPU中，
1.3 torch.set_device(1) 指定第二块（逻辑）GPU为 default，则后续tensor.cuda()迁移到第二块（逻辑）GPU中

2. .to(device)   windows中常用
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
2.1 单个GPU
    model.to(device), func.to(device), data.to(device)
2.1 多个GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = '2, 3, 4, 5'
    device_ids = list(range(torch.cuda.device_count()))  # [0, 1, 2, 3]
    model = torch.nn.DataParallel(model, device_ids=device_ids)
    
    model.to(device), func.to(device), data.to(device)  # 迁移到default中，即第一块（逻辑0）GPU中，实际卡号2GPU中

注意：将model迁移到GPU中，只会将__init__中的有self前缀的”属性“和”函数“放到GPU上，
其他的”数据“和”函数“如果也需要迁移到GPU中，需要格外的.cuda()
"""


# torch.nn.DataParallel方法
"""
model = model.cuda() 
device_ids = [0, 1]  # id为0和1的两块显卡
model = torch.nn.DataParallel(model, device_ids=device_ids)

device_ids = [0, 1]
model = torch.nn.DataParallel(model, device_ids=device_ids).cuda()
"""

## https://zhuanlan.zhihu.com/p/347061440


