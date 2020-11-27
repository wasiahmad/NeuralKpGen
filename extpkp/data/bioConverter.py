import os
import argparse
import json

from tqdm import tqdm
from pathlib import Path
from multiprocessing import Pool

import spacy
import subprocess
from collections import Counter
from nltk.stem.porter import *
from nltk.tokenize import wordpunct_tokenize

stemmer = PorterStemmer()


class SpacyTokenizer(object):

    def __init__(self, **kwargs):
        model = kwargs.get('model', 'en')
        nlp_kwargs = {'parser': False, 'tagger': False, 'entity': False}
        self.nlp = spacy.load(model, **nlp_kwargs)

    def tokenize(self, text):
        doc = self.nlp(text)
        return [token.text for token in doc]

    @property
    def vocab(self):
        return None


class WhiteSpaceTokenizer(object):

    def __init__(self):
        pass

    def tokenize(self, text):
        return text.split()

    @property
    def vocab(self):
        return None


class MultiprocessingTokenizer(object):

    def __init__(self, args):
        self.args = args
        if self.args['tokenizer'] == 'SpacyTokenizer':
            self.tokenizer = SpacyTokenizer(model='en_core_web_sm')
        elif self.args['tokenizer'] == 'WhiteSpace':
            self.tokenizer = WhiteSpaceTokenizer()
        else:
            raise ValueError('Unknown tokenizer type!')

    def initializer(self):
        pass

    def tokenize(self, text):
        tokens = self.tokenizer.tokenize(text)
        return ' '.join(tokens)

    def process(self, example):
        title = example['title'].strip().lower()
        abstract = example['abstract'].strip().lower()

        if self.args['kp_separator']:
            keywords = example['keyword'].lower().split(self.args['kp_separator'])
            keywords = [kp.strip() for kp in keywords]
            present_keywords, absent_keywords = separate_present_absent(
                title + ' ' + abstract, keywords
            )
        else:
            present_keywords = example['present_keywords']
            # absent_keywords = example['absent_keywords']

        # filtering empty keyphrases
        present_keywords = [pkp for pkp in present_keywords if pkp]
        assert len(present_keywords) >= 1
        # absent_keywords = [akp for akp in absent_keywords if akp]

        if self.args['replace_digit_tokenizer']:
            title = fn_replace_digits(title, tokenizer=self.args['replace_digit_tokenizer'])
            abstract = fn_replace_digits(abstract, tokenizer=self.args['replace_digit_tokenizer'])
            present_keywords = [fn_replace_digits(pkp, tokenizer=self.args['replace_digit_tokenizer'])
                                for pkp in present_keywords]
            # absent_keywords = [fn_replace_digits(akp, tokenizer=self.args['replace_digit_tokenizer'])
            #                    for akp in absent_keywords]

        title_tokenized = self.tokenize(title)
        abstract_tokenized = self.tokenize(abstract)
        paragraph = (title_tokenized + ' ' + abstract_tokenized).split()
        paragraph = paragraph[:self.args['max_src_len']]

        pkp_tokenized = [self.tokenize(pkp) for pkp in present_keywords]
        # akp_tokenized = [self.tokenize(akp) for akp in absent_keywords]

        return {
            'id': example['id'],
            'paragraph': paragraph,
            'pkp_tokenized': pkp_tokenized,
            'labels': compute_bio_tags(paragraph, pkp_tokenized)
        }


def count_file_lines(file_path):
    """
    Counts the number of lines in a file using wc utility.
    :param file_path: path to file
    :return: int, no of lines
    """
    num = subprocess.check_output(['wc', '-l', file_path])
    num = num.decode('utf-8').strip().split(' ')
    return int(num[0])


def stem_word_list(word_list):
    return [stemmer.stem(w.strip().lower()) for w in word_list]


def stem_text(text):
    return ' '.join(stem_word_list(text.split()))


def fn_replace_digits(text, tokenizer='wordpunct'):
    out_tokens = []
    in_tokens = wordpunct_tokenize(text) \
        if tokenizer == 'wordpunct' else text.split()
    for tok in in_tokens:
        if re.match('^\d+$', tok):
            out_tokens.append('<digit>')
        else:
            out_tokens.append(tok)
    return ' '.join(out_tokens)


def separate_present_absent(source_text, keyphrases):
    present_kps = []
    absent_kps = []
    stemmed_source = stem_text(source_text)
    for kp in keyphrases:
        stemmed_kp = stem_text(kp)
        if stemmed_kp in stemmed_source:
            present_kps.append(kp)
        else:
            absent_kps.append(kp)

    return present_kps, absent_kps


def compute_bio_tags(para_tokens, present_kps):
    present_kps = [stem_text(pkp) for pkp in present_kps]
    para_tokens = stem_word_list(para_tokens)
    paragraph = ' '.join(para_tokens)
    bio_tags = ['O'] * len(para_tokens)
    for pkp in present_kps:
        if pkp in paragraph:
            pkp_tokens = pkp.split()
            for j in range(0, len(para_tokens) - len(pkp_tokens) + 1):
                if pkp_tokens == para_tokens[j:j + len(pkp_tokens)]:
                    bio_tags[j] = 'B'
                    if len(pkp_tokens) > 1:
                        bio_tags[j + 1:j + len(pkp_tokens)] = ['I'] * (len(pkp_tokens) - 1)

    return bio_tags


def load_data(filename, dataset_name):
    data = []
    # for KP20k and cross-domain datasets
    if dataset_name in ['KP20k', 'inspec', 'krapivin', 'semeval', 'nus']:
        if not os.path.exists(filename[0]):
            return []
        with open(filename[0]) as f1, open(filename[1]) as f2:
            for source, target in tqdm(zip(f1, f2), total=count_file_lines(filename[0])):
                source = source.strip()
                target = target.strip()
                if not source:
                    continue
                if not target:
                    continue

                src_parts = source.split('<eos>', 1)
                assert len(src_parts) == 2
                title = src_parts[0].strip()
                abstract = src_parts[1].strip()

                keywords = target.split('<peos>')
                assert len(keywords) == 2
                present_keywords = [kp.strip() for kp in keywords[0].split(';') if kp]
                # absent_keywords = [kp.strip() for kp in keywords[1].split(';') if kp]
                if len(present_keywords) == 0:
                    continue
                ex = {
                    'id': len(data),
                    'title': title,
                    'abstract': abstract,
                    'present_keywords': present_keywords,
                    # 'absent_keywords': absent_keywords
                }
                data.append(ex)
        print('Dataset loaded from %s and %s.' % (filename[0], filename[1]))
    else:
        if not os.path.exists(filename):
            return []
        with open(filename) as f:
            for line in tqdm(f, total=count_file_lines(filename)):
                ex = json.loads(line)
                if 'id' not in ex:
                    ex['id'] = len(data)
                if 'keywords' in ex:
                    ex['keyword'] = ex['keywords']
                    ex.pop('keywords')

                data.append(ex)
        print('Dataset loaded from %s.' % filename)
    return data


def main(config, TOK):
    pool = Pool(config.workers, initializer=TOK.initializer)

    train_dataset = []
    dataset = load_data(config.train, config.dataset)
    if dataset:
        with tqdm(total=len(dataset), desc='Processing') as pbar:
            for i, ex in enumerate(pool.imap(TOK.process, dataset, 100)):
                pbar.update()
                train_dataset.append(ex)
        with open(os.path.join(config.out_dir, 'train.txt'), 'w', encoding='utf-8') as fw:
            for ex in train_dataset:
                fw.write(
                    json.dumps({'source': ex['paragraph'], 'target': ex['labels']})
                    + '\n'
                )

    valid_dataset = []
    dataset = load_data(config.valid, config.dataset)
    if dataset:
        with tqdm(total=len(dataset), desc='Processing') as pbar:
            for i, ex in enumerate(pool.imap(TOK.process, dataset, 100)):
                pbar.update()
                valid_dataset.append(ex)
        with open(os.path.join(config.out_dir, 'valid.txt'), 'w', encoding='utf-8') as fw:
            for ex in valid_dataset:
                fw.write(
                    json.dumps({'source': ex['paragraph'], 'target': ex['labels']})
                    + '\n'
                )

    test_dataset = []
    dataset = load_data(config.test, config.dataset)
    if dataset:
        with tqdm(total=len(dataset), desc='Processing') as pbar:
            for i, ex in enumerate(pool.imap(TOK.process, dataset, 100)):
                pbar.update()
                test_dataset.append(ex)
        with open(os.path.join(config.out_dir, 'test.txt'), 'w', encoding='utf-8') as fw:
            for ex in test_dataset:
                fw.write(
                    json.dumps({'source': ex['paragraph'], 'target': ex['labels']})
                    + '\n'
                )
        with open(os.path.join(config.out_dir, 'test.source'), 'w', encoding='utf-8') as fw1, \
                open(os.path.join(config.out_dir, 'test.target'), 'w', encoding='utf-8') as fw2:
            for ex in test_dataset:
                fw1.write(' '.join(ex['paragraph']) + '\n')
                fw2.write(';'.join(ex['pkp_tokenized']) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='prepare.py',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-data_dir', required=True,
                        help='Directory where the source files are located')
    parser.add_argument('-out_dir', required=True,
                        help='Directory where the output files will be saved')
    parser.add_argument('-tokenizer', default='SpacyTokenizer',
                        choices=['SpacyTokenizer', 'WhiteSpace'])
    parser.add_argument('-dataset', required=True)
    parser.add_argument('-max_src_len', type=int, default=384)
    parser.add_argument('-workers', type=int, default=20)

    opt = parser.parse_args()

    if not os.path.exists(opt.data_dir):
        raise FileNotFoundError

    Path(opt.out_dir).mkdir(parents=True, exist_ok=True)

    options = dict()
    options['tokenizer'] = opt.tokenizer
    options['max_src_len'] = opt.max_src_len
    options['replace_digit_tokenizer'] = 'wordpunct'
    options['kp_separator'] = ';'

    if opt.dataset == 'KPTimes':
        opt.train = os.path.join(opt.data_dir, 'KPTimes.train.jsonl')
        opt.valid = os.path.join(opt.data_dir, 'KPTimes.valid.jsonl')
        opt.test = os.path.join(opt.data_dir, 'KPTimes.test.jsonl')

    if opt.dataset == 'OAGK':
        options['kp_separator'] = ','
        opt.train = os.path.join(opt.data_dir, 'oagk_train.txt')
        opt.valid = os.path.join(opt.data_dir, 'oagk_val.txt')
        opt.test = os.path.join(opt.data_dir, 'oagk_test.txt')

    if opt.dataset == 'KP20k':
        options['replace_digit_tokenizer'] = None
        options['kp_separator'] = None
        opt.train = (os.path.join(opt.data_dir, 'train_src.txt'),
                     os.path.join(opt.data_dir, 'train_trg.txt'))
        opt.valid = (os.path.join(opt.data_dir, 'valid_src.txt'),
                     os.path.join(opt.data_dir, 'valid_trg.txt'))
        opt.test = (os.path.join(opt.data_dir, 'test_src.txt'),
                    os.path.join(opt.data_dir, 'test_trg.txt'))

    if opt.dataset in ['inspec', 'krapivin', 'semeval', 'nus']:
        options['replace_digit_tokenizer'] = None
        options['kp_separator'] = None
        opt.form_vocab = False
        opt.train = ('', '')
        opt.valid = ('', '')
        opt.test = (
            os.path.join(opt.data_dir, 'word_{}_testing_context.txt'.format(opt.dataset)),
            os.path.join(opt.data_dir, 'word_{}_testing_allkeywords.txt'.format(opt.dataset))
        )

    TOK = MultiprocessingTokenizer(options)
    main(opt, TOK)
