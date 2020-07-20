import logging
import json
import torch
import linecache
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, Sampler
from deepkp.objects import Source, Target, Keyphrase
from deepkp.inputters import constants
from deepkp.utils.misc import count_file_lines

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------------------
# Data loading
# ------------------------------------------------------------------------------


def load_data(text_line, vocab, max_src_len, max_keywords, test_split=False):
    """Load examples from preprocessed file. One example per line, JSON encoded."""

    ex = json.loads(text_line.strip())
    src = Source()
    tgt = Target()

    keyphrases = []
    for kp_text, kp_tokenized in zip(ex['present_kps']['text'],
                                     ex['present_kps']['tokenized']):
        pkp = Keyphrase()
        pkp.text = kp_text
        pkp.tokens = kp_tokenized.split()
        pkp.present = True
        keyphrases.append(pkp)

    for kp_text, kp_tokenized in zip(ex['absent_kps']['text'],
                                     ex['absent_kps']['tokenized']):
        akp = Keyphrase()
        akp.text = kp_text
        akp.tokens = kp_tokenized.split()
        akp.present = False
        keyphrases.append(akp)

    if not test_split:
        keyphrases = keyphrases[:max_keywords]

    tgt.keyphrases = keyphrases
    tgt.form_tokens()
    tgt.vectorize(vocab)  # it is not necessary

    src.title = ex['title']['text']
    src.text = ex['abstract']['text']
    src_tokens = ex['title']['tokenized'].split() + \
                 ex['abstract']['tokenized'].split()
    src.tokens = src_tokens[:max_src_len]
    src.vectorize(vocab)

    return {
        'id': ex['id'],
        'source': src,
        'target': tgt
    }


def load_dataset(filename, vocab, max_src_len, max_keywords,
                 test_split=False, max_examples=-1):
    examples = []
    with open(filename) as f:
        for line in tqdm(f, total=count_file_lines(filename)):
            ex = load_data(line, vocab, max_src_len, max_keywords, test_split)
            examples.append(ex)
            if len(examples) == max_examples:
                break

    return examples


# ------------------------------------------------------------------------------
# Dataset
# ------------------------------------------------------------------------------


class KeyphraseDataset(Dataset):
    def __init__(self, filename, model, lazy_load=False, **kwargs):
        self._model = model
        self.max_src_len = kwargs['max_src_len']
        self.max_keywords = kwargs['max_keywords']
        self.is_test = kwargs.get('test_split', False)
        self._filename = filename
        self._examples = []
        self._total_data = count_file_lines(filename)
        max_examples = kwargs.get('max_examples', -1)
        if max_examples != -1 and self._total_data > max_examples:
            self._total_data = max_examples
        self._lazy_load = lazy_load
        if not lazy_load:
            self._examples = load_dataset(
                filename, self._model.word_dict,
                max_src_len=self.max_src_len,
                max_keywords=self.max_keywords,
                test_split=self.is_test,
                max_examples=self._total_data
            )
            assert len(self._examples) == self._total_data

    def __len__(self):
        return self._total_data

    def __getitem__(self, index):
        if self._lazy_load:
            line = linecache.getline(self._filename, index + 1)
            example = load_data(line, self._model.word_dict,
                                max_src_len=self.max_src_len,
                                max_keywords=self.max_keywords,
                                test_split=self.is_test)
        else:
            example = self._examples[index]
        return self.vectorize(example)

    def vectorize(self, ex):
        ex['target'].form_tokens(choice='all',
                                 bos=constants.BOS_WORD,
                                 eos=constants.EOS_WORD)
        ex['target'].vectorize(self._model.word_dict)
        return ex

    def lengths(self):
        if self._examples:
            return [(len(ex['source'].tokens), len(ex['target'].tokens))
                    for ex in self._examples]
        else:
            raise NotImplementedError


class SortedBatchSampler(Sampler):
    def __init__(self, lengths, batch_size, shuffle=True):
        self.lengths = lengths
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        lengths = np.array(
            [(-l[0], -l[1], np.random.random()) for l in self.lengths],
            dtype=[('l1', np.int_), ('l2', np.int_), ('rand', np.float_)]
        )
        indices = np.argsort(lengths, order=('l1', 'l2', 'rand'))
        batches = [indices[i:i + self.batch_size]
                   for i in range(0, len(indices), self.batch_size)]
        if self.shuffle:
            np.random.shuffle(batches)
        return iter([i for batch in batches for i in batch])

    def __len__(self):
        return len(self.lengths)


# ------------------------------------------------------------------------------
# COLLATE FUNCTION
# ------------------------------------------------------------------------------


def batchify(batch):
    """Gather a batch of individual examples into one batch."""

    out_dict = dict()
    out_dict['batch_size'] = len(batch)
    out_dict['ids'] = [ex['id'] for ex in batch]
    out_dict['source'] = [ex['source'] for ex in batch]
    out_dict['target'] = [ex['target'] for ex in batch]

    # --------- Prepare tensors ---------
    max_src_len = max([len(ex['source']) for ex in batch])
    max_tgt_len = max([len(ex['target']) for ex in batch])
    source_len = torch.zeros(out_dict['batch_size'], dtype=torch.long)
    source_rep = torch.zeros(out_dict['batch_size'], max_src_len, dtype=torch.long)
    target_len = torch.zeros(out_dict['batch_size'], dtype=torch.long)
    target_rep = torch.zeros(out_dict['batch_size'], max_tgt_len, dtype=torch.long)
    source_maps = []
    src_vocabs = []
    alignments = []
    for i in range(out_dict['batch_size']):
        input_ids = batch[i]['source'].input_ids
        vocab = batch[i]['source'].vocab
        src_vocabs.append(vocab)
        source_len[i] = len(input_ids)
        source_rep[i, :len(input_ids)].copy_(torch.tensor(input_ids))
        # Mapping source tokens to indices in the dynamic dict.
        src_map = torch.tensor([vocab[w] for w in batch[i]['source'].tokens])
        source_maps.append(src_map)

        input_ids = batch[i]['target'].input_ids
        target_len[i] = len(input_ids)
        target_rep[i, :len(input_ids)].copy_(torch.tensor(input_ids))
        mask = torch.tensor([vocab[w] for w in batch[i]['target'].tokens])
        alignments.append(mask)

    out_dict['source_len'] = source_len
    out_dict['source_rep'] = source_rep
    out_dict['target_len'] = target_len
    out_dict['target_rep'] = target_rep
    out_dict['src_vocab'] = src_vocabs
    out_dict['src_map'] = source_maps
    out_dict['alignment'] = alignments

    return out_dict
