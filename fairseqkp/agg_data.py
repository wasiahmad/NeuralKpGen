import sys

sys.path.insert(0, '..')

import os
import json
from tqdm import tqdm
from pathlib import Path
from deepkp.inputters import constants

DATA_DIR = '../data/'


def process(infile, outdir, split):
    Path(outdir).mkdir(parents=True, exist_ok=True)
    with open(infile, "r", encoding='utf-8') as fin, \
            open(os.path.join(outdir, '{}.source'.format(split)), 'w', encoding='utf-8') as fsrc, \
            open(os.path.join(outdir, '{}.target'.format(split)), 'w', encoding='utf-8') as ftgt:
        for line in tqdm(fin):
            ex = json.loads(line)
            source = ex['title']['text'] + ' {} '.format(constants.TITLE_SEP) + ex['abstract']['text']
            kps = ex['present_kps']['text'] + ex['absent_kps']['text']
            target = ' ; '.join([t for t in kps if t])
            if len(source) > 0 and len(target) > 0:
                fsrc.write(source + '\n')
                ftgt.write(target + '\n')


def copy_vocab(srcdir, outdir):
    vocab = dict()
    with open(os.path.join(srcdir, 'vocab.txt'), "r") as fin, \
            open(os.path.join(outdir, 'dict.txt'), 'w', encoding='utf-8') as fout:
        vocab['[PAD]'] = len(vocab)
        vocab['[UNK]'] = len(vocab)
        fout.write('[PAD]' + ' ' + str(vocab['[PAD]']) + '\n')
        fout.write('[UNK]' + ' ' + str(vocab['[UNK]']) + '\n')
        for idx, line in enumerate(fin):
            word = line.strip()
            if word not in vocab:
                vocab[word] = len(vocab)
                fout.write(word + ' ' + str(vocab[word]) + '\n')


if __name__ == '__main__':
    process(os.path.join(DATA_DIR, 'scikp/processed/kp20k/train.json'), 'data/kp20k', 'train')
    process(os.path.join(DATA_DIR, 'scikp/processed/kp20k/valid.json'), 'data/kp20k', 'valid')
    process(os.path.join(DATA_DIR, 'scikp/processed/kp20k/test.json'), 'data/kp20k', 'test')
    copy_vocab(os.path.join(DATA_DIR, 'scikp/processed/kp20k'), 'data/kp20k')

    process(os.path.join(DATA_DIR, 'kptimes/processed/train.json'), 'data/kptimes', 'train')
    process(os.path.join(DATA_DIR, 'kptimes/processed/valid.json'), 'data/kptimes', 'valid')
    process(os.path.join(DATA_DIR, 'kptimes/processed/test.json'), 'data/kptimes', 'test')
    copy_vocab(os.path.join(DATA_DIR, 'kptimes/processed'), 'data/kptimes')

    process(os.path.join(DATA_DIR, 'scikp/processed/inspec/test.json'), 'data/inspec', 'test')
    process(os.path.join(DATA_DIR, 'scikp/processed/nus/test.json'), 'data/nus', 'test')
    process(os.path.join(DATA_DIR, 'scikp/processed/krapivin/test.json'), 'data/krapivin', 'test')
    process(os.path.join(DATA_DIR, 'scikp/processed/semeval/test.json'), 'data/semeval', 'test')
