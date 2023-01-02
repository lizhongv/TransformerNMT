import os
import sentencepiece as spm
import torch.nn as nn
import copy
import torch
import numpy as np
import logging


def chinese_tokenizer_load():
    sp_chn = spm.SentencePieceProcessor()
    sp_chn.Load('{}.model'.format("./tokenizer/chn"))
    return sp_chn


def english_tokenizer_load():
    sp_eng = spm.SentencePieceProcessor()
    sp_eng.Load('{}.model'.format("./tokenizer/eng"))
    return sp_eng


def clones(module, N):
    """克隆模型块，克隆的模型块参数不共享"""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def subsequent_mask(size):
    """Mask out subsequent positions."""
    attn_shape = (1, size, size)
    mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(mask) == 0


# TODO  日志改写 https://blog.csdn.net/stellar_liu/article/details/118089901
# https://blog.csdn.net/u011417820/article/details/112861970
def set_logger(log_path):
    """Set the logger to log info in terminal and file `log_path`.
    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.
    Example:
    ```
    logging.info("Starting training...")
    ```
    Args:
        log_path: (string) where to log
    """
    if os.path.exists(log_path) is True:
        os.remove(log_path)

    # 1. 创建日志对象
    logger = logging.getLogger()
    # 2. 设置日志等级 critical, error, warning, info, debug, notset
    logger.setLevel(logging.INFO)

    # 3. 解决日志重复输出问题
    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)

    return logger
