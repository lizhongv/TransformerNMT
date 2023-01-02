import torch
import torch.nn as nn
import numpy as np
import copy
import warnings

import config

# function
from train import train, test, translate
from utils import english_tokenizer_load
from data_loader import load_data

from lib.criterion import LabelSmoothing
from lib.optimizer import get_std_opt, NoamOpt

from model.attention import MultiHeadedAttention
from model.position_wise_feedforward import PositionwiseFeedForward
from model.embedding import PositionalEncoding, Embeddings
from model.transformer import Transformer
from model.encoder import Encoder, EncoderLayer
from model.decoder import Decoder, DecoderLayer
from model.generator import Generator


# TODO model GPU
def make_model(src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
    c = copy.deepcopy

    attn = MultiHeadedAttention(h, d_model).cuda()
    ff = PositionwiseFeedForward(d_model, d_ff, dropout).cuda()
    position = PositionalEncoding(d_model, dropout).cuda()

    model = Transformer(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout).cuda(), N).cuda(),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout).cuda(), N).cuda(),
        nn.Sequential(Embeddings(d_model, src_vocab).cuda(), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab).cuda(), c(position)),
        Generator(d_model, tgt_vocab)
    ).cuda()

    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            # 这里初始化采用的是nn.init.xavier_uniform
            nn.init.xavier_uniform_(p)
    return model.cuda()


def run():
    # 加载数据迭代器
    train_dataloader, dev_dataloader,  test_dataloader = load_data()

    # 初始化模型
    model = make_model(config.src_vocab_size,
                       config.tgt_vocab_size,
                       config.n_layers,
                       config.d_model,
                       config.d_ff,
                       config.n_heads,
                       config.dropout)

    # 损失函数
    if config.use_smoothing:
        criterion = LabelSmoothing(size=config.tgt_vocab_size, padding_idx=config.padding_idx, smoothing=0.1)
        criterion.cuda()  # 其中参数在反向传播中需要更新
    else:
        criterion = torch.nn.CrossEntropyLoss(ignore_index=0, reduction='sum')

    # 优化器
    if config.use_noamopt:
        optimizer = get_std_opt(model)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)

    # 训练
    if config.type == "train":
        config.logger.info(" start train".rjust(35, '>'))
        # nn.DataParallel 并行方式
        model_par = torch.nn.DataParallel(model, device_ids=config.device_ids)
        if model_par.module is model:
            print("model_par.module is model!")
        train(train_dataloader, dev_dataloader, model, model_par, criterion, optimizer)
        config.logger.info(" finished train".rjust(35, '<'))

    # 测试
    elif config.type == "test":
        config.logger.info(" start test".rjust(35, '>'))
        test(test_dataloader, model, criterion)
        config.logger.info(" finished test".rjust(35, '<'))

    else:
        print("Error: please select type within [‘train’, ‘test’]")


def check_opt():
    """check learning rate changes"""
    import numpy as np
    import matplotlib.pyplot as plt
    model = make_model(config.src_vocab_size, config.tgt_vocab_size, config.n_layers,
                       config.d_model, config.d_ff, config.n_heads, config.dropout)
    opt = get_std_opt(model)
    # Three settings of the lrate hyperparameters.
    opts = [opt,
            NoamOpt(512, 1, 20000, None),
            NoamOpt(256, 1, 10000, None)]
    plt.plot(np.arange(1, 50000), [[opt.rate(i) for opt in opts] for i in range(1, 50000)])
    plt.legend(["512:10000", "512:20000", "256:10000"])
    plt.show()


def one_sentence_translate(sent, beam_search=True):
    # 初始化模型
    model = make_model(config.src_vocab_size, config.tgt_vocab_size, config.n_layers,
                       config.d_model, config.d_ff, config.n_heads, config.dropout)

    BOS = english_tokenizer_load().bos_id()  # 2
    EOS = english_tokenizer_load().eos_id()  # 3

    src_tokens = [[BOS] + english_tokenizer_load().EncodeAsIds(sent) + [EOS]]
    batch_input = torch.LongTensor(np.array(src_tokens)).cuda()

    print(" translate ".center(50, '-'))
    translate(batch_input, model, use_beam=beam_search)


def translate_example():
    """单句翻译示例"""
    print(' input sentence'.rjust(35, '>'))
    sent = "But Howard Hughes’s success as a film producer and airline owner made him one of the richest Americans to emerge during the first half of the twentieth century."

    # tgt: 近期的政策对策很明确：把最低工资提升到足以一个全职工人及其家庭免于贫困的水平，扩大对无子女劳动者的工资所得税减免。
    one_sentence_translate(sent, beam_search=True)


if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    # run()
    translate_example()
