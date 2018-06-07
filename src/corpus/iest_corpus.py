import os

# from profilehooks import profile

from ..utils.io import load_or_create
from .batch_iterator import BatchIterator
from .lang import Lang

from .. import config


class BaseCorpus(object):
    def __init__(self, paths_dict, mode='train', use_chars=True,
                 force_reload=False, train_data_proportion=1.0,
                 dev_data_proportion=1.0, batch_size=64,
                 shuffle_batches=False, batch_first=True):

        self.paths = paths_dict
        self.mode = mode

        self.use_chars = use_chars

        self.force_reload = force_reload

        self.train_data_proportion = train_data_proportion
        self.dev_data_proportion = dev_data_proportion

        self.batch_size = batch_size
        self.shuffle_batches = shuffle_batches
        self.batch_first = batch_first


class IESTCorpus(BaseCorpus):

    # @profile(immediate=True)
    def __init__(self, *args, **kwargs):
        super(IESTCorpus, self).__init__(config.corpora_dict['iest'],
                                         *args, **kwargs)

        train_sents = open(self.paths['train']).readlines()

        # This assumes the data comes nicely separated by spaces. That's the
        # task of the tokenizer who should be called elsewhere
        self.train_sents = [s.rstrip().split() for s in train_sents]

        dev_sents = open(self.paths['dev']).readlines()
        self.dev_sents = [s.rstrip().split() for s in dev_sents]

        lang_pickle_path = os.path.join(config.CACHE_PATH, 'lang.pkl')
        self.lang = load_or_create(lang_pickle_path,
                                   Lang,
                                   self.train_sents,
                                   force_reload=self.force_reload)

        self.label2id = {key: value for key, value in config.LABEL2ID.items()}

        self.train_examples = self._create_examples(self.train_sents, mode='train')
        self.dev_examples = self._create_examples(self.dev_sents, mode='dev')

        self.train_batches = BatchIterator(self.train_examples,
                                           self.batch_size,
                                           data_proportion=self.train_data_proportion,
                                           shuffle=True,
                                           batch_first=self.batch_first,
                                           use_chars=self.use_chars)

        self.dev_batches = BatchIterator(self.dev_examples,
                                         self.batch_size,
                                         data_proportion=self.dev_data_proportion,
                                         shuffle=False,
                                         batch_first=self.batch_first,
                                         use_chars=self.use_chars)

    def _create_examples(self, sents, mode):
        """
        sents: list of strings
        mode: (string) train, dev or test

        return:
            examples: a list containing dicts representing each example
        """

        allowed_modes = ['train', 'dev', 'test']
        if mode not in allowed_modes:
            raise ValueError(f'Mode not recognized, try one of {allowed_modes}')

        id_sents_pickle_path = os.path.join(config.CACHE_PATH, mode + '.pkl')
        id_sents = load_or_create(id_sents_pickle_path,
                                  self.lang.sents2ids,
                                  sents,
                                  force_reload=self.force_reload)

        chars_pickle_path = os.path.join(config.CACHE_PATH,
                                         mode + '_chars.pkl')
        char_id_sents = load_or_create(chars_pickle_path,
                                       self.lang.sents2char_ids,
                                       sents,
                                       force_reload=self.force_reload)

        labels = open(config.LABEL_PATHS[mode]).readlines()
        labels = [l.rstrip() for l in labels]
        id_labels = [self.label2id[label] for label in labels]

        ids = range(len(id_sents))

        examples = zip(ids,
                       id_sents,
                       char_id_sents,
                       id_labels)

        examples = [{'id': ex[0],
                     'sequence': ex[1],
                     'char_sequence': ex[2],
                     'label': ex[3]} for ex in examples]

        return examples
