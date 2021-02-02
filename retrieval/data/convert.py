import json
import argparse


def main(args):
    # {'id': '', 'title': '', 'keyword': '', 'abstract': ''}
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
                kps = ';'.join(pkps + akps)
                obj = {
                    'id': idx,
                    'title': title,
                    'abstract': abstract,
                    'keyword': kps
                }
                idx += 1
                fw.write(json.dumps(obj) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='convert.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '-src_file', required=True, help='Source file'
    )
    parser.add_argument(
        '-tgt_file', required=True, help='Target file'
    )
    parser.add_argument(
        '-out_file', required=True, help='Output file path'
    )
    args = parser.parse_args()
    main(args)
