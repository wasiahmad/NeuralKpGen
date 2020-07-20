from deepkp.inputters import constants
from deepkp.inputters.vocabulary import Vocabulary


class Source(object):

    def __init__(self, _id=None):
        self._id = _id
        self._text = None
        self._title = None
        self._tokens = []
        self._input_ids = []
        self.vocab = None  # required for Copy Attention

    @property
    def id(self) -> str:
        return self._id

    @property
    def text(self) -> str:
        return self._text

    @text.setter
    def text(self, param: str) -> None:
        assert isinstance(param, str), 'Source.text should be a String'
        self._text = param

    @property
    def title(self) -> str:
        return self._title

    @title.setter
    def title(self, param: str) -> None:
        assert isinstance(param, str), 'Source.title should be a String'
        self._title = param

    @property
    def tokens(self) -> list:
        return self._tokens

    @tokens.setter
    def tokens(self, param: list) -> None:
        if not isinstance(param, list):
            raise TypeError('Source.tokens must be a list')
        self._tokens = param
        self.form_vocab()

    def form_vocab(self) -> None:
        self.vocab = Vocabulary(no_special_token=True)
        self.vocab.add_tokens(self.tokens)

    def vectorize(self, vocab) -> list:
        if not self._input_ids:
            self._input_ids = [vocab[w] for w in self.tokens]
        return self._input_ids

    @property
    def input_ids(self) -> list:
        return self._input_ids

    @input_ids.setter
    def input_ids(self, param: list) -> None:
        self._input_ids = param

    def __len__(self):
        return len(self._tokens)
