# Needed for avoiding kernel restart in jupyter

# COMMANDS FOR DOWNLOADING MODEL
# import os
# os.system("python -m spacy download en_core_web_lg")

import numpy as np

import spacy
try: import en_core_web_lg 
except ModuleNotFoundError: pass

def get_preprocessor():
    nlp = None
    
    # needed to avoid kernel restart to use spacy.load('en_core_web_lg')
    try: nlp = en_core_web_lg.load()
    except ModuleNotFoundError: nlp = spacy.load('en') 

    nlp.tokenizer.add_special_case("<EOS>", [{spacy.symbols.ORTH: "<EOS>"}])
    nlp.tokenizer.add_special_case("<BOS>", [{spacy.symbols.ORTH: "<BOS>"}])
    nlp.tokenizer.add_special_case("<PAD>", [{spacy.symbols.ORTH: "<PAD>"}])
     
    v = np.ones_like(nlp.vocab['<BOS>'].vector) * -1
    nlp.vocab.set_vector("<PAD>", v)

    return nlp