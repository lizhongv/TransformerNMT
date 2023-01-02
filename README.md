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

