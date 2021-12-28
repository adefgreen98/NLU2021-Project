import json
import torch
from random import shuffle
from collections import defaultdict


class ATISDataset(torch.utils.data.Dataset):
    
    # NOTE: we can safely suppose that entities are ordered for each sentences both in train and test

    # Tag for tokens without entity
    __sos_token = '<SOS>'
    __eos_token = '<EOS>'
    __padding_token = '<PAD>'
    __empty_ent = 'O'

    __embedder = None

    def __init__(self, path, embedder):
        """
        :param path: path of the .json file containing the dataset.
        """
        self._raw = json.load(open(path))["rasa_nlu_data"]["common_examples"]

        self._statistics = None

        # embedder with vocabulary and tokenizer (for vector conversion)
        if ATISDataset.__embedder is None:
            ATISDataset.__embedder = embedder

        # retrieve unique entities
        entities = set()
        for data in self._raw:
            for ent in data["entities"]: entities.add(ent["entity"])
        entities.add(ATISDataset.__empty_ent)
        entities = list(entities)

        # reverse dict of tag indices
        self.tag2idx = {k: i for i, k in enumerate(entities)}

        # iterable list of items in the dataset
        self._seq_items = self.preprocess_items()

        # populates statistics with preprocessed items
        self._statistics = self.get_statistics()


    def __getitem__(self, index):
        return self._seq_items[index]

    def __len__(self):
        return len(self._seq_items)
    
    
    def __iter__(self):
        yield from self._seq_items
    
    
    @staticmethod
    def collate_ce(batch):
        """
        Returns a batch of sentences padded to be the same length. This is specific for the **cross-entropy** loss function:
        labels will be retrieved as tensors of tag indices for the relative sentence, stacked in a batch.

        *NOTE*: this method expects to receive sentences **alredy padded** to equal length AND already transformed in tensors.
        Inputs:
        :param batch: a list of samples of dimension batch_size (specified by the external DataLoader)
        Returns:
        :param x: a tensor of stacked tensors, each one a sentence (as stacked tensor of embedded tokens)
        :param y: a tensor of stacked tensors for each sentence, containing the label **index** for each token. 
        """
        x = torch.stack([b[0] for b in batch], dim=0)
        y = torch.stack([b[1] for b in batch], dim=0)
        return x,y


    def get_statistics(self):
        """
        Returns a dictionary of statistics for the dataset. Currently it contains:
        - 'entities': a list of all possible tags 
        - 'ent_counts': a dictionary associating concept names to their count in the dataset
        - 'size': number of sentences in the dataset
        - 'len_max': maximum sentence length (used also for padding)
        - 'len_avg': average sentence length
        """
        if self._statistics is None: self._populate_statistics()
        return self._statistics


    def get_labels(self):
        return self.tag2idx


    def get_max_sent_length(self):
        if self._statistics is None: return max([len(d["text"].split()) for d in self._raw])
        else:
            return self._statistics["len_max"]


    def get_avg_sent_length(self):
        if self._statistics is None:
            return sum([len(d["text"].split()) for d in self._raw]) / len(self._raw)
        else:
            self._statistics["len_avg"]

    def preprocess_items(self):
        """
        Transforms the whole dataset sample by sample to match the required format (i.e. aligned tensor of stacked tokens & list of entity tags).

        In future, may also perform other operations (such as augmentation or other transformations).
        """
        l = [None for i in range(len(self._raw))]
        for i in range(len(self._raw)):
            snt, lab = self._raw[i]["text"], self._raw[i]["entities"] 
            x, y = ATISDataset._align_label(snt, lab)
            x, y = ATISDataset._pad_and_preprocess(x, y, self.get_max_sent_length())
            x = torch.stack([torch.tensor(ATISDataset.__embedder.vocab[word].vector) for word in x])
            y = torch.tensor([self.tag2idx[tag] for tag in y]) 
            l[i] = (x, y)
        return l

    @staticmethod
    def preprocess_single_sentence(sent, pad_len):
        """
        Transforms one single sentence in vector format, acceptable by a model.
        Used for model inference.
        """
        x, _ = ATISDataset._align_label(sent, [])
        x, _ = ATISDataset._pad_and_preprocess(x, [], pad_len)
        return torch.stack([torch.tensor(ATISDataset.__embedder.vocab[word].vector) for word in x])


    #### Private methods ####

    def _populate_statistics(self):
        if self._statistics is not None:
            print("Warning: already full statistics")
        else:
            self._statistics = {
                "entities": None,
                "ent_counts": defaultdict(int),
                "size": len(self._raw),
                "len_max": self.get_max_sent_length(),
                "len_avg": self.get_avg_sent_length()
            }
            
            self._statistics["entities"] = list(self.tag2idx.keys())

            for _, label in self._seq_items:
                for ent_idx in label:
                    self._statistics["ent_counts"][self._statistics["entities"][ent_idx]] += 1
        return


    @staticmethod
    def _align_label(sentence, label):
        """ 
        Transforms 1 single sample in form of 2 aligned lists: a token-by-token sentence and a list of entities, 
        with none entities tagged with the tag specified statically in this method's class.
        """

        res_sent = []
        res_ent = []
        ent_idx = 0
        tmp_tok = ""
        tmp_ent = ATISDataset.__empty_ent
        last = None
        for i in range(len(sentence)):
            if ent_idx == len(label): # no more entities
                last = i
                break

            # space or last character --> flush current token and entity
            if sentence[i] == " " or i == len(sentence) - 1:
                if i == len(sentence) - 1:
                    tmp_tok = tmp_tok + sentence[i]
                res_sent.append(tmp_tok)
                tmp_tok = ""
                res_ent.append(tmp_ent)
                tmp_ent = ATISDataset.__empty_ent
            else:
                tmp_tok = tmp_tok + sentence[i]

            if label[ent_idx]["start"] <= i < label[ent_idx]["end"]:
                tmp_ent = label[ent_idx]["entity"]
            elif i == label[ent_idx]["end"]:
                ent_idx += 1 
                # not changing entity: supposing we are doing it when a space is met
        if last is not None:
            coda = sentence[last:].split()
            res_sent.extend(coda)
            res_ent.extend([ATISDataset.__empty_ent] * len(coda))
        
        assert len(res_sent) == len(res_ent)
        return res_sent, res_ent


    @staticmethod
    def _pad_and_preprocess(sentence, label, pad_size, right_pad=True):
        """
        :param sentence: a list containing the tokenized sentence before padding
        :param label: a list containing the tokenwise entity tags
        :param pad_size: dimension that the sentence must reach (without considering added tags!)
        :param right_pad: bool determining whether to pad right (True) or left (False)
        Returns:
        :param sentence: list of tokens
        :param label: list of concept tags for each token, aligned with the sentence
        """
        to_pad = pad_size - len(sentence)
        if right_pad:
            #right padding
            sentence = [ATISDataset.__sos_token] + sentence + [ATISDataset.__eos_token] + [ATISDataset.__padding_token] * to_pad
            label = [ATISDataset.__empty_ent] + label + [ATISDataset.__empty_ent] + [ATISDataset.__empty_ent] * to_pad
        else: 
            #left padding
            sentence = [ATISDataset.__padding_token] * to_pad + [ATISDataset.__sos_token] + sentence + [ATISDataset.__eos_token]
            label = [ATISDataset.__empty_ent] * to_pad + [ATISDataset.__empty_ent] + label + [ATISDataset.__empty_ent]
        return sentence, label


class ATISSubset(torch.utils.data.Subset):
    """
    A class for easily subsetting an ATISDataset without losing the possibility 
    of keeping a custom collate function (used by the torch DataLoader).
    """
    def collate_ce(self, batch):
        """Collate function for CrossEntropyLoss."""
        return self.dataset.collate_ce(batch)


def split_dataset(dataset, valid_ratio=0.1):
    """
    Involved in splitting training set and validation set from the original dataset.
    Inputs:
    :param ratio: percentage of samples to put in validation set
    Outputs:
    :param training_set: torch.utils.data.Subset containing samples for training
    :param valid_set: same as training_set, but with samples for evaluation
    """
    _idx = list(range(len(dataset)))
    _last = round(len(_idx) * (1 - valid_ratio))
    shuffle(_idx)
    train = ATISSubset(dataset, _idx[:_last])
    valid = ATISSubset(dataset, _idx[_last:])
    return train, valid