import torch
import sacrebleu
from tqdm import tqdm

import config

from beam_decoder import beam_search
from greedy_decoder import batch_greedy_decode
from utils import chinese_tokenizer_load
from lib.loss import SimpleLossCompute, MultiGPULossCompute


def run_epoch(data, model, loss_compute):
    total_tokens = 0.
    total_loss = 0.

    for batch in tqdm(data):
        out = model(batch.src, batch.trg, batch.src_mask, batch.trg_mask)
        loss = loss_compute(out, batch.trg_y, batch.ntokens)

        total_loss += loss
        total_tokens += batch.ntokens
    return total_loss / total_tokens


def train(train_data, dev_data, model, model_par, criterion, optimizer):
    """
    Args:
        train_data:
        dev_data:
        model:
        model_par: nn.Parallerl 分布式训练模型
        criterion:
        optimizer:

    Returns:
    """
    best_bleu_score = 0.0
    early_stop = config.early_stop  # 模型得分连续多次低于最优则提前停止

    for epoch in range(1, config.epoch_num + 1):
        # 模型训练
        model.train()
        """
        train_loss = run_epoch(train_data, model_par,
                               MultiGPULossCompute(model.generator, criterion, config.device_id, optimizer))
        """
        train_loss = run_epoch(train_data, model_par,
                               SimpleLossCompute(model.generator, criterion, optimizer))
        config.logger.info("Epoch: {}, loss: {}".format(epoch, train_loss))

        # 模型验证
        model.eval()
        """
        dev_loss = run_epoch(dev_data, model_par,
                             MultiGPULossCompute(model.generator, criterion, config.device_id, None))
        """
        dev_loss = run_epoch(dev_data, model_par,
                             SimpleLossCompute(model.generator, criterion, optimizer))

        # 评价函数, 验证集
        bleu_score = evaluate(dev_data, model)
        config.logger.info('Epoch: {}, Dev loss: {}, Bleu Score: {}'.format(epoch, dev_loss, bleu_score))

        # 如果当前epoch的模型在dev集上的loss优于之前记录的最优loss则保存当前模型，并更新最优loss值
        if bleu_score > best_bleu_score:
            # 保存模型
            torch.save(model.state_dict(), config.model_path)

            best_bleu_score = bleu_score
            early_stop = config.early_stop
            config.logger.info("-------- Save Best Model! --------")
        else:
            early_stop -= 1
            config.logger.info("Early Stop Left: {}".format(early_stop))

        if early_stop == 0:
            config.logger.info("-------- Early Stop! --------")
            break


def evaluate(data, model, mode='dev', use_beam=True):
    """ 评价函数 """
    sp_chn = chinese_tokenizer_load()
    trg = []
    res = []

    with torch.no_grad():
        # 在data的英文数据长度上遍历下标
        for batch in tqdm(data):
            # 对应的中文句子
            cn_sent = batch.trg_text
            src = batch.src
            src_mask = (src != 0).unsqueeze(-2)

            # 集束搜索
            if use_beam:
                decode_result, _ = beam_search(model, src, src_mask,
                                               config.max_len,
                                               config.padding_idx,
                                               config.bos_idx,
                                               config.eos_idx,
                                               config.beam_size)
            # 贪婪搜索
            else:
                decode_result = batch_greedy_decode(model, src, src_mask,
                                                    max_len=config.max_len)

            decode_result = [h[0] for h in decode_result]
            translation = [sp_chn.decode_ids(_s) for _s in decode_result]
            trg.extend(cn_sent)
            res.extend(translation)

    # 若是测试，则将翻译结果写入文件
    if mode == 'test':
        with open(config.output_path, "w") as fp:
            for i in range(len(trg)):
                line = "idx:" + str(i) + '\n' + trg[i] + '\n' + res[i] + '\n'
                fp.write(line)

    trg = [trg]
    bleu = sacrebleu.corpus_bleu(res, trg, tokenize='zh')
    return float(bleu.score)


def test(data, model, criterion):
    with torch.no_grad():
        # 加载模型
        model.load_state_dict(torch.load(config.model_path))
        # DP方式
        model_par = torch.nn.DataParallel(model, device_ids=config.device_ids).cuda()

        # 模型验证
        model.eval()
        """
         test_loss = run_epoch(data, model_par,
                              MultiGPULossCompute(model.generator, criterion, config.device_id, None))
        """
        test_loss = run_epoch(data, model_par,
                              SimpleLossCompute(model.generator, criterion))

        # 评价函数
        bleu_score = evaluate(data, model, 'test')
        config.logger.info('Test loss: {},  Bleu Score: {}'.format(test_loss, bleu_score))


def translate(src, model, use_beam=True):
    """用训练好的模型进行预测单句，打印模型翻译结果"""
    sp_chn = chinese_tokenizer_load()
    with torch.no_grad():
        # 加载模型
        model.load_state_dict(torch.load(config.model_path))

        model.eval()
        src_mask = (src != 0).unsqueeze(-2)
        if use_beam:
            decode_result, _ = beam_search(model, src, src_mask, config.max_len,
                                           config.padding_idx, config.bos_idx, config.eos_idx,
                                           config.beam_size)
            decode_result = [h[0] for h in decode_result]
        else:
            decode_result = batch_greedy_decode(model, src, src_mask, max_len=config.max_len)
        translation = [sp_chn.decode_ids(_s) for _s in decode_result]

        print(" translate result".rjust(35, '<'))
        print(translation[0])
