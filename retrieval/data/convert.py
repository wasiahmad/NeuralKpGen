import json
import argparse


def scikp(args):
    idx = 0
    with open(args.out_file, 'w', encoding='utf8') as fw:
        with open(args.src_file) as f1, open(args.tgt_file) as f2:
            for source, target in zip(f1, f2):
                source = source.strip()
                target = target.strip()
                if len(source) == 0 or len(target) == 0:
                    continue
                title, abstract = [s.strip() for s in source.split('<eos>')]
                if len(title) == 0 or len(abstract) == 0:
                    continue
                pkps, akps = [kp.strip().split(';') for kp in target.split('<peos>')]
                pkps = [kp for kp in pkps if kp]
                akps = [kp for kp in akps if kp]
                if len(pkps) == 0 and len(akps) == 0:
                    continue
                ex_idx = ''
                if args.dataset:
                    ex_idx += args.dataset + '.'
                if args.split:
                    ex_idx += args.split + '.'
                ex_idx += str(idx)
                obj = {
                    'id': ex_idx,
                    'title': title,
                    'abstract': abstract,
                    'present': pkps,
                    'absent': akps
                }
                idx += 1
                fw.write(json.dumps(obj) + '\n')


def kptimes(args):
    idx = 0
    with open(args.out_file, 'w', encoding='utf8') as fw:
        with open(args.input_file) as f:
            for line in f:
                ex = json.loads(line.strip())
                if len(ex['title']) == 0 or len(ex['abstract']) == 0:
                    continue
                keyphrases = ex['keyword'].split(';')
                pkps, akps = [], []
                text = (ex['title'] + ' ' + ex['abstract']).strip()
                for kp in keyphrases:
                    if kp in text:
                        pkps.append(kp)
                    else:
                        akps.append(kp)
                obj = {
                    'id': ex['id'],
                    'title': ex['title'],
                    'abstract': ex['abstract'],
                    'present': pkps,
                    'absent': akps
                }
                idx += 1
                fw.write(json.dumps(obj) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='convert.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('-input_file', default=None, help='Input json file')
    parser.add_argument('-src_file', default=None, help='Source *.txt file')
    parser.add_argument('-tgt_file', default=None, help='Target *.txt file')
    parser.add_argument('-out_file', required=True, help='Output file path')
    parser.add_argument('-dataset', default=None, help='Dataset name',
                        choices=['kp20k', 'inspec', 'nus', 'krapivin', 'semeval', 'kptimes'])
    parser.add_argument('-split', default=None, help='Dataset name')
    args = parser.parse_args()
    if args.dataset == 'kptimes':
        assert args.input_file is not None
        kptimes(args)
    else:
        assert args.src_file is not None
        assert args.tgt_file is not None
        scikp(args)
