"""
# To Do

## General
* add an LSTM character-wise encoder for word embeddings
* test LSTM-embedder against word2vec / Spacy embedding vectors
* test instability with / without teacher forcing
* torchtext **does** provide pretrained embeddings https://pytorch.org/text/stable/vocab.html#vectors
* Embedding required for `Decoder` too?? --> Needed to embed previous output
* Test if Spacy embedding vectors include the `<unk>` token

## Code specific
* solve the `nan` accuracy issue (due to zero items of a specific category in validation or training) --> maybe use sigmoid / squashing number of false positives?
* remove get_model dependency on 'labels' and write a 'set_labels' method
* add tqdm for epochs

"""