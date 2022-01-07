"""
# To Do

## General
Main:
* add **beam search** at test time
* add **attention**
* add **weighted loss** to see if it boosts accuracy
* see if **bidirectionality** boosts metrics
* implement f1

Embeddings:
* add an LSTM character-wise encoder for word embeddings
* torchtext **does** provide pretrained embeddings https://pytorch.org/text/stable/vocab.html#vectors
* see what happens with **unfrozen** embeddings

Others
* test instability with / without teacher forcing
* understand which output is the correct one for Decoder output

## Code specific
* solve the `nan` accuracy issue (due to zero items of a specific category in validation or training) --> maybe use sigmoid / squashing number of false positives?
* remove get_model dependency on 'labels' and write a 'set_labels' method
* add tqdm for epochs

"""