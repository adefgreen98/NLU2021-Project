import pickle
import torch
from random import shuffle
from collections import defaultdict


class ATISDataset(torch.utils.data.Dataset):
    

    def __init__(self, path):
        """
        :param path: path of the .json file containing the dataset.
        """
        _raw, _dicts = pickle.load(open(path, 'rb'))

        self._tokens_dict = {v: k for k,v in _dicts["token_ids"].items()}
        self._tag_dict = {v: k for k,v in _dicts["slot_ids"].items()}
        self._data = [(q, lab) for q, lab in zip(_raw["query"], _raw["slot_labels"])]

    
    def __len__(self):
        return len(self._data)


    def __getitem__(self, index):
        sent, lab = self._data[index]
        sent = [self._tokens_dict[tok] for tok in sent]
        lab = [self._tag_dict[tag] for tag in lab]
        return (sent, lab)
    

    def get_entities(self):
        return list(self._tag_dict.values())
    

    # TODO: get statistics