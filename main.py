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
from loss import *


"""# Main method"""

def main(cfg, **kwargs):

    # DATASET
    dataset = get_dataset(kwargs["train_path"])
    train_set, valid_set = split_dataset(dataset, valid_ratio=cfg["valid_ratio"], rnd=False)
    
    train_dataloader = get_dataloader(train_set, batch_size=cfg["batch_size"], shuffle=True, collate_type='ce')
    valid_dataloader = get_dataloader(valid_set, batch_size=cfg["batch_size"], collate_type='ce')
    
    # MODEL & OTHER OBJECTS
    embedder = Embedder(kwargs["train_path"])
    net = get_model(embedder, cfg["model"], cfg["hidden_size"], device=get_device())
    optimizer = get_optimizer(net, cfg["learning_rate"], cfg["optimizer"])
    loss_fn = get_loss(cfg["loss"], embedder)
    
    # TRAINING
    metrics = train(cfg["nr_epochs"], net, train_dataloader, optimizer, loss_fn, valid_dl=valid_dataloader, greedy_save=True)

    # SAVING
    _path = 'tests'
    if kwargs["save"]:
        _path = save_model(net, cfg)
        save_training(metrics, _path)
    
    # TESTING
    test_dataset = get_dataset(kwargs["test_path"])
    
    test(net, test_dataset, save_path=_path)
    test_beam_search(net, test_dataset, save_path=_path)



def produce_configurations(params):
    param_names = list(params.keys())
    configurations = product(*list(params.values()))

    for cfg in configurations:
        yield {k: v for k, v in zip(param_names, cfg)}



parameters = {
    "valid_ratio": [0.2],
    "batch_size": [64],
    "model": ["gru"],
    "loss": ["masked_ce"],
    "optimizer": ["adam"],
    "hidden_size": [200],
    "learning_rate": [1e-3],
    "nr_epochs": [15],
}


if __name__ == '__main__':

    for cfg in produce_configurations(parameters):
        print()
        print("---> Configuration: <---", *[str(k) + ": " + str(v) for k,v in cfg.items()], sep='\n')
        main(cfg, train_path = "iob_atis/atis.train.pkl", test_path = "iob_atis/atis.test.pkl", save=True)
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print()



