# Needed for avoiding kernel restart in jupyter

# COMMANDS FOR DOWNLOADING MODEL
# import os
# os.system("python -m spacy download en_core_web_lg")

import pickle
import numpy as np

import spacy
try: import en_core_web_lg 
except ModuleNotFoundError: pass

import torch
import torchtext

from collections import defaultdict

class TorchEmbedding(object):

    def __init__(self, dataset_path='iob_atis/atis.train.pkl', learn=False, vec_size=300):
        assert vec_size in [50, 100, 200, 300]
        tmpvocab = torchtext.vocab.GloVe(name='6B', dim=vec_size)
        self.word2idx = tmpvocab.stoi.copy()
        tmpvectors = tmpvocab.vectors

        with open(dataset_path, 'rb') as f:
            _, dataset_dicts = pickle.load(f) 
            vocab = list(dataset_dicts["token_ids"].keys())

        # adding unknown words in dataset
        to_append = []
        for word in vocab:
            try: _ = self.word2idx[word]
            except KeyError:
                to_append.append(torch.randn(vec_size) - 0.5)
                self.word2idx[word] = len(self.word2idx)
        
        # adding padding words
        self.word2idx['<BOS>'] = len(self.word2idx)
        to_append.append(torch.zeros(vec_size, dtype=tmpvectors.dtype))
        
        self.word2idx['<EOS>'] = len(self.word2idx)
        to_append.append(torch.zeros(vec_size,dtype=tmpvectors.dtype))

        self.word2idx['<PAD>'] = len(self.word2idx)
        to_append.append(torch.ones(vec_size, dtype=tmpvectors.dtype) * -1)

        # appending new vectors to old ones
        tmpvectors = torch.cat((tmpvectors, torch.stack(to_append)))

        self.emb = torch.nn.Embedding.from_pretrained(tmpvectors, freeze=(not learn)).cuda()
        

    def __getitem__(self, word):
        return self.emb(torch.LongTensor([self.word2idx[word]]).cuda())

    def get_sent(self, sent):
        return self.emb(torch.LongTensor([self.word2idx[word] for word in sent]).cuda())

    def get_size(self):
        return self.emb(torch.LongTensor([0]).cuda()).shape[-1]


class SpacyEmbedding(object):

    def __init__(self):
        nlp = None
        # needed to avoid kernel restart to use spacy.load('en_core_web_lg')
        try: nlp = en_core_web_lg.load()
        except ModuleNotFoundError: nlp = spacy.load('en') 

        nlp.tokenizer.add_special_case("<EOS>", [{spacy.symbols.ORTH: "<EOS>"}])
        nlp.tokenizer.add_special_case("<BOS>", [{spacy.symbols.ORTH: "<BOS>"}])
        nlp.tokenizer.add_special_case("<PAD>", [{spacy.symbols.ORTH: "<PAD>"}])
        
        v = np.ones_like(nlp.vocab['<BOS>'].vector) * -1
        nlp.vocab.set_vector("<PAD>", v)

        self.nlp = nlp.vocab
    
    def __getitem__(self, word):
        return torch.tensor(self.nlp[word].vector)
    
    def get_batch(self, sent):
        pass

    def get_size(self):
        return self.nlp.vectors_length


def get_preprocessor(method):
    if method == 'spacy': return SpacyEmbedding()
    elif method == 'glove': return TorchEmbedding(learn=True)
    else: raise ValueError("unsupported embedding method")