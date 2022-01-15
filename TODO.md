"""
# To Do

## General
Main:
* add random (?) baseline
* add CRF baseline (also CRF + embeddings)
* what other CRF features can be used?
* understand if needed **unknown** words management
* understand what input pytorch is feeding into decoder (sentence or previous label)
* see if **bidirectionality** boosts metrics
* add **attention**
* add **weighted loss** to see if it boosts accuracy

Graphing:
* add option for naming 'not_found' parameter
* add distinction between selected attribute and selected configurations of other excluded attributes

Embeddings:

Others 
* find a way to balance statistical summations
* retrieve number of parameters
* get reference for beam search
* what was the paper for sampling methods in GLP? **nucleus sampling?**
* pca for reducting dimensionality of initial embeddings
* test instability with / without teacher forcing

Dataset Metrics:
* number of samples
* labels distribution (top50 ?)
* size of tag set
* 

## Code specific
* solve the `nan` accuracy issue (due to zero items of a specific category in validation or training) --> maybe use sigmoid / squashing number of false positives?

"""