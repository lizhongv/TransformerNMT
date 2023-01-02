import json

if __name__ == "__main__":
    files = ['train', 'dev', 'test']
    ch_path = 'corpus.ch'
    en_path = 'corpus.en'
    ch_lines = []
    en_lines = []

    for file in files:
        with open('./json/' + file + '.json', 'r') as f:
            corpus = json.load(f)  # 读取`文件句柄`
            for item in corpus:
                en_lines.append(item[0] + '\n')
                ch_lines.append(item[1] + '\n')

    with open(ch_path, "w") as fch:
        fch.writelines(ch_lines)
    print(" Get corpus.ch! ".center(50, '-'))

    with open(en_path, "w") as fen:
        fen.writelines(en_lines)
    print(" Get corpus.en! ".center(50, '-'))

    # lines of Chinese: 252777
    print("lines of Chinese: ", len(ch_lines))
    # lines of English: 252777
    print("lines of English: ", len(en_lines))
    print(" Get Corpus! ".center(50, '-'))
