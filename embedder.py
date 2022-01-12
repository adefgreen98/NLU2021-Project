import pickle
import torch 
import torchtext

from nlp_init import get_preprocessor


class Embedder():

    __padding_token = '<PAD>'
    __begin_tok = '<BOS>'
    __end_tok = '<EOS>'
    __empty_ent = 'O'
 
    def __init__(self, entities_path=None, embedding_method='glove', ext_vocab=None, dataset=None):
        
        # Initialize word embeddings
        self.embedding_method = embedding_method

        self.ext_vocab = ext_vocab
        if self.ext_vocab is None: 
            self.ext_vocab = get_preprocessor(embedding_method) 

        # Initialize entities
        assert (entities_path is None and dataset is not None) or (entities_path is not None and dataset is None) 
        if dataset is not None:
            entities = dataset.get_entities()
        else:
            with open(entities_path, 'rb') as f:
                entities = list(pickle.load(f)[1]["slot_ids"].keys())
        
        entities.append(Embedder.__padding_token)

        self.tag2idx = {k: i for i, k in enumerate(entities)}
        self.idx2tag = list(self.tag2idx.keys()) #needed for model inference
    

    def __call__(self, x, y=None, _batch=False):
        """
        Converts a sentence into a tensor of vectorized tokens. If y is provided, it also tensorizes the label.
        """
        if not _batch:
            if y is None:
                return self.preprocess_single_sentence(x)
            else:
                x,y = self.get_test_sentence(x, y)
                x = self.ext_vocab.get_sent(x)
                y = torch.stack([torch.tensor([self.tag2idx[tag] for tag in y], dtype=torch.uint8)])
                return x,y
        else:
            x = torch.stack([self.ext_vocab.get_sent(b) for b in x], dim=0)
            y = torch.stack([torch.tensor([self.tag2idx[tag] for tag in b], dtype=torch.uint8) for b in y], dim=0)
            return x,y


    @staticmethod
    def collate_ce(batch):
        """
        Returns a batch of sentences padded to be the same length. This is specific for the **cross-entropy** loss function:
        labels will be retrieved as tensors of tag indices for the relative sentence, stacked in a batch.

        Inputs:
        :param batch: a list of samples of dimension batch_size (specified by the external DataLoader)
        Returns:
        :param x: a tensor of stacked tensors, each one a sentence (as stacked tensor of embedded tokens)
        :param y: a tensor of stacked tensors for each sentence, containing the label **index** for each token. 
        """

        batch_max_len = max([len(sample[0]) for sample in batch])

        res_batch_x = []
        res_batch_y = []
        for sample in batch:
            x, y = Embedder._apply_padding(sample[0], sample[1], batch_max_len)
            res_batch_x.append(x)
            res_batch_y.append(y)
        
        return res_batch_x, res_batch_y


    def get_padding_token_index(self):
        return self.tag2idx[Embedder.__padding_token]
    

    def get_labels_dict(self):
        return self.tag2idx


    def get_entities(self):
        return list(filter(lambda x: x != Embedder.__padding_token, list(self.tag2idx.keys())))


    def get_embedding_size(self):
        return self.ext_vocab.get_size()


    def preprocess_single_sentence(self, sent=None, pad_len=None):
        """
        Transforms one single sentence in vector format, acceptable by a model.
        Used for model inference.
        """
        x = Embedder._preprocess_test_sentence(sent, pad_len)
        return self.ext_vocab.get_sent(x)


    def get_test_sentence(self, sent, lab):
        """
        Retrieves one sample (sentence, tags) from the dataset to be used in testing, WITHOUT embedding sample & label.
        """
        return self._apply_padding(sent, lab, len(sent))
    

    # Private methods


    @staticmethod
    def _apply_padding(sentence, label, pad_size, right_pad=True):
        """
        NOTE: entities associated with padding tokens are padding tokens themselves, so that
        they can be recognized and masked out.
        :param sentence: a list containing the tokenized sentence before padding
        :param label: a list containing the tokenwise entity tags
        :param pad_size: dimension that the sentence must reach (without considering added tags!)
        :param right_pad: bool determining whether to pad right (True) or left (False)
        Returns:
        :param sentence: list of tokens
        :param label: list of concept tags for each token, aligned with the sentence
        """

        to_pad = pad_size - len(sentence)

        # converts BOS, EOS in the specfied format
        sentence[0] = Embedder.__begin_tok
        sentence[-1] = Embedder.__end_tok
        
        if len(label) > 0:
            label[0] = Embedder.__empty_ent
            label[-1] = Embedder.__empty_ent

        if right_pad:
            #right padding
            sentence = sentence + [Embedder.__padding_token] * to_pad
            label = label + [Embedder.__padding_token] * to_pad
        else: 
            #left padding
            sentence = [Embedder.__padding_token] * to_pad + sentence
            label = [Embedder.__padding_token] * to_pad + label
        return sentence, label


    @staticmethod
    def _preprocess_test_sentence(sent, pad_len=None):
        """
        Preprocesses (= align and apply padding) a single sentence for the use-case of inference during testing, i.e. without labels.
        """

        if pad_len is None: pad_len = len(sent)
        x, _ = Embedder._apply_padding(sent, [], pad_len)
        return x
