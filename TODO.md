"""
# To Do

## General
Main:
* implement f1
* learnable torchtext weigths
* add **attention**
* add **weighted loss** to see if it boosts accuracy
* see if **bidirectionality** boosts metrics

Embeddings:
* add an LSTM character-wise encoder for word embeddings
* torchtext **does** provide pretrained embeddings https://pytorch.org/text/stable/vocab.html#vectors
* see what happens with **unfrozen** embeddings

Others
* test instability with / without teacher forcing

## Code specific
* download evaluator script from https://github.com/sighsmile/conlleval/blob/master/conlleval.py 
* solve the `nan` accuracy issue (due to zero items of a specific category in validation or training) --> maybe use sigmoid / squashing number of false positives?
* add tqdm for epochs

"""