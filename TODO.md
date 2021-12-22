"""# To Do

* add an LSTM character-wise encoder for word embeddings
* test LSTM-embedder against word2vec / Spacy embedding vectors
* test instability with / without teacher forcing
* torchtext **does** provide pretrained embeddings https://pytorch.org/text/stable/vocab.html#vectors
* Embedding required for `Decoder` too?? --> Needed to embed previous output
* Test if Spacy embedding vectors include the `<unk>` token

# Datasets
* IMDB and others from torchtext
* PennTreebank, CONLL, others from the course
* What is the most correct format to load data??
* Some info on legacy vs modern pytorch at https://colab.research.google.com/github/pytorch/text/blob/master/examples/legacy_tutorial/migration_tutorial.ipynb#scrollTo=zCvxeLbYW3I

# Papers
* Gobbi, Stepanov, Riccardi http://disi.unitn.it/~riccardi/papers2/Clicit18-concept-tagging.pdf
* Search for *slot filling* and *entity extraction*
* Bechét, Raymond (2018) https://hal.inria.fr/hal-01835425/document 
* https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6998838 complete DL review
*
"""