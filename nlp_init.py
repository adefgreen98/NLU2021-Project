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


def get_embedder(name, vec_size=300, dataset_path='iob_atis/atis.train.pkl'):
    if name == 'glove':
        assert vec_size in [50, 100, 200, 300]
        tmpvocab = torchtext.vocab.GloVe(name='6B', dim=vec_size)

        with open(dataset_path, 'rb') as f:
            _, dataset_dicts = pickle.load(f) 
            vocab = list(dataset_dicts["token_ids"].keys())

        # selecting only relevant embeddings
        to_append = []
        word2idx = {}
        for word in vocab:
            word2idx[word] = len(to_append)
            try: 
                i = tmpvocab.stoi[word]
                to_append.append(tmpvocab.vectors[i].squeeze())
            except KeyError:
                to_append.append(torch.randn(vec_size) - 0.5)
        
        # adding padding words
        word2idx['<BOS>'] = len(word2idx)
        to_append.append(torch.zeros(vec_size, dtype=tmpvocab.vectors.dtype))
        
        word2idx['<EOS>'] = len(word2idx)
        to_append.append(torch.zeros(vec_size, dtype=tmpvocab.vectors.dtype))

        word2idx['<PAD>'] = len(word2idx)
        to_append.append(torch.ones(vec_size, dtype=tmpvocab.vectors.dtype) * -1)

        # appending new vectors to old ones
        tmpvectors = torch.stack(to_append)

        res = torch.nn.Embedding.from_pretrained(tmpvectors, freeze=False).cuda()
        res.word2idx = word2idx

        def get_sent(self, sent): return self(torch.LongTensor([self.word2idx[word] for word in sent]).cuda())
        def get_vec_size(self): return self.weight.shape[-1]
        res.get_sent = get_sent.__get__(res)
        res.get_vec_size = get_vec_size.__get__(res)

        return res
    elif name == 'spacy':
        nlp = None
        # needed to avoid kernel restart to use spacy.load('en_core_web_lg')
        try: nlp = en_core_web_lg.load()
        except ModuleNotFoundError: nlp = spacy.load('en') 

        nlp.tokenizer.add_special_case("<EOS>", [{spacy.symbols.ORTH: "<EOS>"}])
        nlp.tokenizer.add_special_case("<BOS>", [{spacy.symbols.ORTH: "<BOS>"}])
        nlp.tokenizer.add_special_case("<PAD>", [{spacy.symbols.ORTH: "<PAD>"}])
        
        v = np.ones_like(nlp.vocab['<BOS>'].vector) * -1
        nlp.vocab.set_vector("<PAD>", v)

        def get_sent(self, sent): return torch.stack([torch.tensor(self.vocab[word].vector) for word in sent])
        nlp.get_sent = get_sent.__get__(nlp)
        def get_vec_size(self): return self.vocab.vectors_length
        nlp.get_vec_size = get_vec_size.__get__(nlp)

        return nlp

    else:
        raise NotImplementedError





def get_preprocessor(method):
    if method == 'spacy': return SpacyEmbedding()
    elif method == 'glove': return TorchEmbedding(learn=True)
    else: raise ValueError("unsupported embedding method")