import pickle
import torch
import textwrap

import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

from random import shuffle
from pprint import pprint
from collections import defaultdict
from collections import Counter

def get_tag(iob_s): return iob_s.split('-')[-1] if iob_s != 'O' else None


class ATISDataset(torch.utils.data.Dataset):
    
    def __init__(self, path="iob_atis\\atis.train.pkl"):
        """
        :param path: path of the .json file containing the dataset.
        """
        _raw, _dicts = pickle.load(open(path, 'rb'))

        self.statistics = None
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
    

    def get_statistics(self):

        if self.statistics is None: 
            self.statistics = {}
            
            max_len = {'idx': None, 'val': -1}
            avg_sum = 0
            tag_dist = []
            co_occurrence = []
            nr_tags_per_sentence = []
            iob_lengths = defaultdict(list)
            vocabulary = []
            for i in range(len(self)):
                sent, lab = self[i]
                if len(sent) > max_len['val']:
                    max_len['idx'] = i
                    max_len['val'] = len(sent)
                avg_sum += len(sent)

                for s in sent:
                    vocabulary.append(s)
                
                for i in range(len(lab)):
                    if lab[i] == 'O' or lab[i].startswith('I'): continue
                    elif lab[i].startswith('B'):
                        curr_tag = get_tag(lab[i])
                        curr_len = 1
                        i += 1
                        while lab[i].startswith('I'):
                            i+=1
                            curr_len += 1
                        iob_lengths[curr_tag].append(curr_len)

                full_tags = [get_tag(l) for l in lab if l != 'O']
                tag_dist.extend(full_tags)
                nr_tags_per_sentence.append(len(full_tags))

                unique_tags = sorted(list(set(full_tags)))
                co_occurrence.append("+".join(unique_tags))

            self.statistics['max_length'] = max_len['val'] - 2
            self.statistics['avg_length'] = avg_sum / len(self) - 2
            self.statistics["tag_dist"] = Counter(tag_dist)
            self.statistics['co_occurrence'] = Counter(co_occurrence)
            self.statistics["avg_tags_per_sent"] = torch.tensor(nr_tags_per_sentence).float().mean().item()
            self.statistics["avg_iob_length"] = {k: torch.tensor(iob_lengths[k]).float().mean().item() for k in iob_lengths}
            self.statistics['vocab'] = Counter(vocabulary)
            self.statistics['vocab'].pop('BOS')
            self.statistics['vocab'].pop('EOS')


        return self.statistics

if __name__ == "__main__":
    dataset = ATISDataset("iob_atis\\atis.train.pkl")
    stats = dataset.get_statistics()

    print('Max Length', stats['max_length'])
    print('Average Lenght', stats['avg_length'])
    print('Average Tags per Sentence', stats['avg_tags_per_sent'])

    most_common_size = 20
    co_occur_size = 10
    iob_len_size = most_common_size
    vocab_size = most_common_size
    palette = 'Spectral'
    plt.figure(figsize=(12, 10))
    
    # tag distribution
    ax = plt.subplot(3,2,1)
    ax.grid()
    # ax.set_title(f'Distribution of {most_common_size} most common tags')
    df = pd.DataFrame.from_records([{'name': k, 'count': v} for k,v in dict(stats['tag_dist'].most_common(most_common_size)).items()])
    sns.barplot(data=df, y='name', x='count', ax=ax, palette=palette)
    sns.despine()
    ax.set_xlabel('')
    ax.set_ylabel('')
    
    def change_height(_ax, new_value) :
        for patch in _ax.patches :
            current_height = patch.get_height()
            diff = current_height - new_value
            patch.set_height(new_value)
            patch.set_y(patch.get_y() + diff * .5)

    # vocabulary
    ax = plt.subplot(3,2,3)
    # ax.set_title(f'Most common {vocab_size} words')
    ax.grid()
    df = pd.DataFrame.from_records([{'word': k, 'count': v} for k,v in dict(stats['vocab'].most_common(vocab_size)).items()])
    sns.barplot(data=df, y='word', x='count', ax=ax, palette=palette)
    sns.despine()
    ax.set_xlabel('')
    ax.set_ylabel('')

    # iob length analysis
    ax = plt.subplot(3,2,5)
    ax.grid()
    # ax.set_title(f'IOB length (top {most_common_size})')
    df = pd.DataFrame.from_records(sorted([{'tag': k, 'average IOB length': v} for k,v in stats['avg_iob_length'].items()], key=lambda el: el['average IOB length'], reverse=True)[:most_common_size])
    sns.barplot(data=df, y='tag', x='average IOB length', ax=ax, palette=palette)
    sns.despine()
    ax.set_xlabel('')
    ax.set_ylabel('')

    # co-occurrence of tags
    ax = plt.subplot(1, 2, 2)
    ax.grid()
    # ax.set_title(f'Co-occurrence of tags - most {co_occur_size}')
    df = pd.DataFrame.from_records([{'name': textwrap.fill(k.replace("+", " + "), width=30, break_long_words=False), 'count': v} for k,v in dict(stats['co_occurrence'].most_common(co_occur_size)).items()])
    sns.barplot(data=df, y='name', x='count', ax=ax, palette=palette)
    sns.despine()
    ax.set_xlabel('')
    ax.set_ylabel('')

    # show figures
    plt.tight_layout()
    plt.show() 

