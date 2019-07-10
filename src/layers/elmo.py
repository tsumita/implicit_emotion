import sys

from .. import config
from ..utils.torch import to_var

import torch

sys.path.append(config.ALLENNLP_PATH)

# from allennlp.modules.elmo import Elmo, batch_to_ids
from allennlp.modules.elmo import batch_to_ids
from .elmoformanylangs import Embedder


def sentences_padding(sentences):
    max_sent_len = len(max(sentences, key=len))
    for sent_id, sent in enumerate(sentences):
        sent_len = len(sent)
        sentence = [sent[idx] if idx<sent_len else '' for idx in range(max_sent_len)]
        sentences[sent_id] = sentence
    return sentences


class ElmoWordEncodingLayer(object):
    def __init__(self, **kwargs):
        kwargs.pop('use_cuda')
        kwargs.pop('char_embeddings')
        kwargs.pop('char_hidden_size')
        kwargs.pop('train_char_embeddings')
        kwargs.pop('word_char_aggregation_method')
#         self._embedder = Elmo(config.ELMO_OPTIONS, config.ELMO_WEIGHTS,
#                               num_output_representations=1, **kwargs)
        self._embedder = Embedder(config.ELMO_JP_PATH)
#         self._embedder = self._embedder.cuda()

        # We know the output of ELMo with pre-trained weigths is of size 1024.
        # This would likely be different if we initialized `Elmo` with a custom
        # `module`, but we didn't test this, so for now it will be hardcoded.
        self.embedding_dim = 1024

    def __call__(self, *args):
        """Sents a batch of N sentences represented as list of tokens"""

        # -1 is the raw_sequences element passed in the encode function of the
        # IESTClassifier
        # TODO: make this less hacky
        sents = args[-1]

#         print(sents)
#         char_ids = batch_to_ids(sents)
#         char_ids = to_var(char_ids,
#                           use_cuda=False,
#                           requires_grad=False)
        # returns a dict with keys: elmo_representations (list) and mask (torch.LongTensor)
#         embedded = self._embedder(char_ids)
#         print(char_ids)
        sents = sentences_padding(sents)
        embedded = self._embedder.sents2elmo(sents)
        return torch.Tensor(embedded)

#         embeddings = embedded['elmo_representations'][0]
#         mask = embedded['mask']
#         embeddings = to_var(embeddings,
#                             use_cuda=False,
#                             requires_grad=False)

#         return embeddings, mask
