import random
from deepkp.inputters import constants


class Keyphrase(object):

    def __init__(self):
        self._text = None
        self._tokens = []
        self._present = False

    @property
    def text(self) -> str:
        return self._text

    @text.setter
    def text(self, param: str) -> None:
        self._text = param

    @property
    def tokens(self) -> list:
        return self._tokens

    @tokens.setter
    def tokens(self, param: list) -> None:
        if not isinstance(param, list):
            raise TypeError('Keyphrase.tokens must be a list')
        self._tokens = param

    @property
    def present(self) -> bool:
        return self._present

    @present.setter
    def present(self, param: bool) -> None:
        assert isinstance(param, bool)
        self._present = param

    def __len__(self):
        return len(self.tokens)


class Target(object):

    def __init__(self, _id=None, sep=constants.KP_SEP,
                 psep=constants.PRESENT_EOS):
        self._id = _id
        self._sep = sep
        self._psep = psep
        self._keyphrases = []
        self._tokens = []
        self._input_ids = []

    @property
    def id(self) -> str:
        return self._id

    @property
    def sep(self) -> str:
        return self._sep

    @property
    def psep(self) -> str:
        return self._psep

    @property
    def keyphrases(self) -> list:
        return self._keyphrases

    @property
    def present_keyphrases(self) -> list:
        return [kp for kp in self._keyphrases if kp.present]

    @property
    def absent_keyphrases(self) -> list:
        return [kp for kp in self._keyphrases if not kp.present]

    @keyphrases.setter
    def keyphrases(self, param: list) -> None:
        assert all(isinstance(x, Keyphrase) for x in param)
        self._keyphrases = param

    @property
    def present_text(self) -> str:
        """Target text is formed by concatenating the
        keyphrase text using separator delimiter."""
        return (' %s ' % self.sep).join([kp.text for kp in self.present_keyphrases])

    @property
    def absent_text(self) -> str:
        """Target text is formed by concatenating the
        keyphrase text using separator delimiter."""
        return (' %s ' % self.sep).join([kp.text for kp in self.absent_keyphrases])

    @property
    def text(self) -> str:
        """Target text is formed by concatenating the
        keyphrase text using separator delimiter."""
        return self.present_text + (' %s ' % self.psep) + self.absent_text

    @property
    def tokens(self) -> list:
        return self._tokens

    def _add_keyphrase_tokens(self, keyphrases, shuffle):
        if len(keyphrases) == 0:
            return
        indices = list(range(len(keyphrases)))
        if shuffle:
            random.shuffle(indices)
        for i, idx in enumerate(indices):
            self._tokens += keyphrases[idx].tokens
            if i < len(keyphrases) - 1:
                self._tokens += [self.sep]

    def form_tokens(self, shuffle=False, choice='all', bos=None, eos=None):
        """List of target tokens is formed by concatenating the keyphrase tokens
        using SEPARATOR delimiter. The special start and end token is also added.
        :param choice: {'all', 'present', 'absent'}
        """
        self._tokens = []
        if choice in ['present', 'all']:
            self._add_keyphrase_tokens(self.present_keyphrases, shuffle)
            self._tokens += [self.psep]
        if choice in ['absent', 'all']:
            self._add_keyphrase_tokens(self.absent_keyphrases, shuffle)

        if bos:
            self._tokens.insert(0, bos)
        if eos:
            self._tokens.append(eos)

    @property
    def sep_mask(self) -> list:
        """A boolean list indicate whether a token is the special SEPARATOR token"""
        return [int(tok == constants.KP_SEP) for tok in self.tokens]

    @property
    def input_ids(self) -> list:
        return self._input_ids

    @input_ids.setter
    def input_ids(self, param: list) -> None:
        self._input_ids = param

    def vectorize(self, vocab) -> list:
        self._input_ids = [vocab[w] for w in self.tokens]
        return self._input_ids

    def __len__(self):
        return len(self.tokens)
