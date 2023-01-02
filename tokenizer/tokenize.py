import sentencepiece as spm  # 分词，训练分词模型，构建词表

# 模型训练 | 四种方式
# 1. unigram    一元分词，分成一个一个汉字
# 2. bpe        字节对编码，先将词分成一个一个字符，然后在词范围内统计词对出现的频次，每次将次数最多的词对保存，直到构成完整的词表
# 3. char       字符型分词
# 4. word       语料需先进行预分词


def train(input_file, vocab_size, model_name, model_type, character_coverage):
    """
    search on https://github.com/google/sentencepiece/blob/master/doc/options.md to learn more about the parameters
    :param input_file: one-sentence-per-line raw corpus file. No need to run tokenizer, normalizer or preprocessor.
                       By default, SentencePiece normalizes the input with Unicode NFKC.
                       You can pass a comma-separated list of files.
    :param vocab_size: vocabulary size, e.g., 8000, 16000, or 32000
    :param model_name: output model name prefix. <model_name>.model and <model_name>.vocab are generated.
    :param model_type: model type. Choose from unigram (default), bpe, char, or word.
                       The input sentence must be pretokenized when using word type.
    :param character_coverage: amount of characters covered by the model, good defaults are: 0.9995 for languages with
                               rich character set like Japanse or Chinese and 1.0 for other languages with
                               small character set.
    """
    spm.SentencePieceTrainer.Train(
        ' '.join([
            f'--input={input_file}',
            f'--model_prefix={model_name}',
            f'--vocab_size={vocab_size}',
            f'--model_type={model_type}',
            f'--character_coverage={character_coverage}',
            f'--pad_id=0',
            f'--unk_id=1',
            f'--bos_id=2',
            f'--eos_id=3',
        ])
    )


def run():
    en_input = '../data/corpus.en'
    en_vocab_size = 32000
    en_model_name = 'eng'
    en_model_type = 'bpe'
    en_character_coverage = 1
    train(en_input, en_vocab_size, en_model_name, en_model_type, en_character_coverage)

    ch_input = '../data/corpus.ch'
    ch_vocab_size = 32000
    ch_model_name = 'chn'
    ch_model_type = 'bpe'
    ch_character_coverage = 0.9995
    train(ch_input, ch_vocab_size, ch_model_name, ch_model_type, ch_character_coverage)


def test():
    sp = spm.SentencePieceProcessor()
    sp.Load("./chn.model")

    text = "美国总统特朗普今日抵达夏威夷。"
    print(" Get Encode ! ".center(50, '-'))
    print(sp.EncodeAsPieces(text))  # ['▁美国总统', '特朗普', '今日', '抵达', '夏威夷', '。']
    print(sp.EncodeAsIds(text))  # [12907, 277, 7419, 7318, 18384, 28724]

    a = [12907, 277, 7419, 7318, 18384, 28724]
    print(" Get Decode ! ".center(50, '-'))
    print(sp.decode_ids(a))  # 美国总统特朗普今日抵达夏威夷。


if __name__ == "__main__":
    # run()
    test()
