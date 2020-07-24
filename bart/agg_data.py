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
            target = ' {} '.format(constants.KP_SEP).join([t for t in kps if t])
            if len(source) > 0 and len(target) > 0:
                fsrc.write(source + '\n')
                ftgt.write(target + '\n')


if __name__ == '__main__':
    process(os.path.join(DATA_DIR, 'scikp/processed/kp20k/train.json'), 'data/kp20k', 'train')
    process(os.path.join(DATA_DIR, 'scikp/processed/kp20k/valid.json'), 'data/kp20k', 'valid')
    process(os.path.join(DATA_DIR, 'scikp/processed/kp20k/test.json'), 'data/kp20k', 'test')
    process(os.path.join(DATA_DIR, 'kptimes/processed/train.json'), 'data/kptimes', 'train')
    process(os.path.join(DATA_DIR, 'kptimes/processed/valid.json'), 'data/kptimes', 'valid')
    process(os.path.join(DATA_DIR, 'kptimes/processed/test.json'), 'data/kptimes', 'test')
    process(os.path.join(DATA_DIR, 'scikp/processed/inspec/test.json'), 'data/inspec', 'test')
    process(os.path.join(DATA_DIR, 'scikp/processed/nus/test.json'), 'data/nus', 'test')
    process(os.path.join(DATA_DIR, 'scikp/processed/krapivin/test.json'), 'data/krapivin', 'test')
    process(os.path.join(DATA_DIR, 'scikp/processed/semeval/test.json'), 'data/semeval', 'test')
