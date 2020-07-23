import sys

sys.path.insert(0, '../../')

import os
import json
from tqdm import tqdm
from pathlib import Path
from deepkp.inputters import constants

DATA_DIR = '../../data/'


def process(infile, outfile):
    out_examples = []
    with open(infile) as fin:
        for line in tqdm(fin):
            ex = json.loads(line)
            source = ex['title']['text'] + ' {} '.format(constants.TITLE_SEP) + ex['abstract']['text']
            kps = ex['present_kps']['text'] + ex['absent_kps']['text']
            target = ';'.join([t for t in kps if t])
            out_examples.append({
                'src': source,
                'tgt': target
            })
    with open(outfile, 'w') as fout:
        fout.write('\n'.join([json.dumps(ex) for ex in out_examples]))


if __name__ == '__main__':
    Path('data').mkdir(parents=True, exist_ok=True)
    process(os.path.join(DATA_DIR, 'scikp/kp20k/train.json'), 'data/kp20k_train.json')
    process(os.path.join(DATA_DIR, 'scikp/kp20k/valid.json'), 'data/kp20k_valid.json')
    process(os.path.join(DATA_DIR, 'scikp/kp20k/test.json'), 'data/kp20k_test.json')
    process(os.path.join(DATA_DIR, 'scikp/inspec/test.json'), 'data/inspec_test.json')
    process(os.path.join(DATA_DIR, 'scikp/nus/test.json'), 'data/nus_test.json')
    process(os.path.join(DATA_DIR, 'scikp/krapivin/test.json'), 'data/krapivin_test.json')
    process(os.path.join(DATA_DIR, 'scikp/semeval/test.json'), 'data/semeval_test.json')
    process(os.path.join(DATA_DIR, 'kptimes/processed/train.json'), 'data/kptimes_train.json')
    process(os.path.join(DATA_DIR, 'kptimes/processed/valid.json'), 'data/kptimes_valid.json')
    process(os.path.join(DATA_DIR, 'kptimes/processed/test.json'), 'data/kptimes_test.json')
