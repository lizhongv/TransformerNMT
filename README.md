## TransformerNMT
This is a `Transformer` based neural machine translation(NMT) model.

## Project Structure
- data
  - json
    - train.json
    - dev.json
    - test.json
  - get_corpus.py  | Extract Chinese and English corpus from `train/dev/test` and save it to `corpus.en` and `corpus.ch`
- lib
  - criterion.py 
  - loss.py
  - optimizer.py 
- model | Implement the Transformer 
- tokenizer
  - chn.model
  - chn.vocab
  - eng.model
  - eng.vocab
  - tokenize.py | Use `sentencepiece.SentencePieceTrainer.Train()` method trains a `BPE` word segmentation model and vocabulary.
- data_loader.py | Data processing. Build the dataset `train/dev/train` from the JSON file, which in turn builds the `iterator`.
- utils.py  
- greedy_decoder.py
- beam_decoder.py
- config.py
- train.py
- main.py

## Corpus
[WMT 2018](https://statmt.org/wmt18/translation-task.html)

| Dataset   | train |  dev |  test |
| :-------: | :-------: | :-------: | :------: |
| size | 176943 | 25278 | 50556 |

## Model 
[harvardnlp/annotated-transformer](https://github.com/harvardnlp/annotated-transformer)

## Requirements
```
pip install -r requirements.txt
```

## Usage
Model parameters can be set in `config.py`. </br>
Adopt `nn.DataParallel` distributed training method, by setting `os.environ['CUDA_VISIBLE_DEVICES']` using multi-GPU training.

## Result
- Train
On the training set, the best model is reached after 32 cycles of training, with a loss of 2.23 on the training set, 4.10 on the validation set, and a BLEU score of 26.07.

- Test
| Beam_size   | 2 |  3 | 4 | 5 |
| :-------: | :-------: | :-------: | :------: | :----------:|
| Bleu | 26.67 | 26.81 | 26.84 | 26.87 |

## Translation
English：
```
But Howard Hughes’s success as a film producer and airline owner made him one of the richest Americans to emerge during the first half of the twentieth century. 
```
ground true:
```
但霍华德·休斯作为电影制片人和航空公司老板的成功使得他跻身20世纪前半叶最富有的美国人行列。 
```
translation:
```
但霍华德·赫伯特作为电影生产商,航空所有者他在20世纪首代美国人出现的最富有美国人之一。 
```
