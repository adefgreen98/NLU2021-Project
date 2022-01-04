# -*- coding: utf-8 -*-
"""preprocess_embed.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1aMxmDzFuFuUJ5Hr23pZCT-pTRjpfWZ15

# General Notes
-  Considering entities as multiple tokens **not stuck together**, each token must be recognized independently from the others
- Scanning sentence left-right: since entities should be ordered we are avoiding assigning same entity to duplicate tokens

# Dependencies
"""

from itertools import product

from utils import *
from nlp_init import get_preprocessor
from loss import *




"""# Main loop"""

def main(**kwargs):
    nlp = get_preprocessor()

    dataset = get_dataset(kwargs["train_path"], nlp)
    labels = dataset.get_labels()
    train_set, valid_set = split_dataset(dataset, valid_ratio=kwargs["valid_ratio"])
    
    train_dataloader = get_dataloader(train_set, batch_size=kwargs["batch_size"], shuffle=False, collate_type='ce')
    eval_dataloader = get_dataloader(valid_set, batch_size=kwargs["batch_size"], collate_type='ce')
    # test_dataloader = get_dataloader(get_dataset(kwargs["test_path"], nlp), batch_size=1)
    
    net = get_model(kwargs["model"], labels, nlp.vocab.vectors_length, kwargs["hidden_size"], device=get_device())
    
    optimizer = get_optimizer(net, kwargs["learning_rate"], kwargs["optimizer"])

    loss_fn = get_loss(kwargs["loss"], dataset)
    
    train(kwargs["nr_epochs"], net, train_dataloader, optimizer, loss_fn, valid_dl=eval_dataloader)

    # building embedder to have just 1 argument, as needed by the model
    # inference_embedder_fn = lambda sent: dataset.preprocess_single_sentence(sent, dataset.get_max_sent_length())
    # net.set_embedder(inference_embedder_fn)
    
    net.set_embedder(dataset.preprocess_single_sentence)

    test(net, valid_set, 10)



def produce_configurations(params):
    param_names = list(params.keys())
    configurations = product(*list(params.values()))

    for cfg in configurations:
        yield {k: v for k, v in zip(param_names, cfg)}



parameters = {
    "valid_ratio": [0.2],
    "batch_size": [64],
    "model": ["gru"],
    "loss": ["ce", "masked_ce"],
    "optimizer": ["adam"],
    "hidden_size": [200],
    "learning_rate": [1e-4],
    "nr_epochs": [20],
}


if __name__ == '__main__':

    for cfg in produce_configurations(parameters):
        print()
        print("---> Configuration: <---", *[str(k) + ": " + str(v) for k,v in cfg.items()], sep='\n')
        main(train_path = "ATIS/train.json", test_path = "ATIS/test.json", save_path = "", **cfg)
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print()



