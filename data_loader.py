import torch
import json
import numpy as np
from torch.autograd import Variable

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from utils import english_tokenizer_load, chinese_tokenizer_load, subsequent_mask

import config


# TODO data GPU and mask
class Batch:
    """Object for holding a batch of data with mask during training."""
    def __init__(self, src_text, trg_text, src, trg=None, pad=0):
        self.src_text = src_text
        self.trg_text = trg_text
        # 加入GPU
        src = src.cuda()
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)  # 非pad处为True，pad处为False

        # 如果输出目标不为空，则需要对decoder要使用到的target句子进行mask
        if trg is not None:
            # 加入GPU
            trg = trg.cuda()
            # decoder要用到的target输入部分
            self.trg = trg[:, :-1]
            # decoder训练时应预测输出的target结果
            self.trg_y = trg[:, 1:]
            self.trg_mask = self.make_std_mask(self.trg, pad)  # 将target输入部分进行attention mask
            self.ntokens = (self.trg_y != pad).data.sum()  # 将应输出的target结果中实际的词数进行统计

    @staticmethod
    def make_std_mask(tgt, pad):
        """Create a mask to hide padding and future words."""
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask


class MTDataset(Dataset):
    def __init__(self, data_path):
        self.out_en_sent, self.out_cn_sent = self.get_dataset(data_path, sort=True)
        self.sp_eng = english_tokenizer_load()
        self.sp_chn = chinese_tokenizer_load()
        self.PAD = self.sp_eng.pad_id()  # 0
        self.BOS = self.sp_eng.bos_id()  # 2
        self.EOS = self.sp_eng.eos_id()  # 3

    @staticmethod
    def len_argsort(seq):
        """传入一系列句子数据(分好词的列表形式)，按照句子长度排序后，返回排序后原来各句子在数据中的索引下标"""
        return sorted(range(len(seq)), key=lambda x: len(seq[x]))

    def get_dataset(self, data_path, sort=False):
        """把中文和英文按照同样的顺序排序, 以英文句子长度排序的(句子下标)顺序为基准"""
        dataset = json.load(open(data_path, 'r'))
        en_sents = []
        cn_sents = []
        for idx, _ in enumerate(dataset):
            en_sents.append(dataset[idx][0])
            cn_sents.append(dataset[idx][1])
        if sort:
            sorted_index = self.len_argsort(en_sents)
            sorted_en_sents = [en_sents[i] for i in sorted_index]
            sorted_cn_sents = [cn_sents[i] for i in sorted_index]
        return sorted_en_sents, sorted_cn_sents

    def __getitem__(self, idx):
        eng_text = self.out_en_sent[idx]
        chn_text = self.out_cn_sent[idx]
        return [eng_text, chn_text]

    def __len__(self):
        return len(self.out_en_sent)

    def collate_fn(self, batch):
        """自定义手动将抽取出的样本堆叠起来的函数"""
        src_text = [x[0] for x in batch]
        tgt_text = [x[1] for x in batch]

        src_tokens = [[self.BOS] + self.sp_eng.EncodeAsIds(sent) + [self.EOS] for sent in src_text]
        tgt_tokens = [[self.BOS] + self.sp_chn.EncodeAsIds(sent) + [self.EOS] for sent in tgt_text]

        # batch 数据对齐, batch_first=True 表示形成 bach_size 在第一维，即形成 [batch_size, max_len_seq]
        batch_input = pad_sequence([torch.LongTensor(np.array(l_)) for l_ in src_tokens],
                                   batch_first=True, padding_value=self.PAD)
        batch_target = pad_sequence([torch.LongTensor(np.array(l_)) for l_ in tgt_tokens],
                                    batch_first=True, padding_value=self.PAD)

        return Batch(src_text, tgt_text, batch_input, batch_target, self.PAD)


# TODO  改写数据准备
def load_data():
    # 构建`数据类`
    train_dataset = MTDataset(config.train_data_path)
    dev_dataset = MTDataset(config.dev_data_path)
    test_dataset = MTDataset(config.test_data_path)
    print("-------- Dataset Build! --------")

    # 构建 `数据迭代器`
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=config.batch_size,
                                  collate_fn=train_dataset.collate_fn)
    dev_dataloader = DataLoader(dev_dataset, shuffle=False, batch_size=config.batch_size,
                                collate_fn=dev_dataset.collate_fn)
    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=config.batch_size,
                                 collate_fn=test_dataset.collate_fn)
    print("-------- Get Dataloader! --------")

    return train_dataloader, dev_dataloader, test_dataloader

# if __name__ == "__main__":
#     load_data()