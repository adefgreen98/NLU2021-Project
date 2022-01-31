# -*- coding: utf-8 -*-

from pprint import pprint
from itertools import product

from utils import *
from loss import *
from statistics import add_results, build_histogram, get_stats_path, set_stats_path
from collections import defaultdict

"""# Main method"""

def main(cfg, **kwargs):

    # DATASET
    dataset = get_dataset(kwargs["train_path"])
    
    train_set, valid_set = split_dataset(dataset, valid_ratio=0.25, rnd=True)
    test_dataset = get_dataset(kwargs["test_path"])
    
    train_dataloader = get_dataloader(train_set, batch_size=cfg["batch_size"], shuffle=True)
    valid_dataloader = get_dataloader(valid_set, batch_size=cfg["batch_size"])
    test_dataloader = get_dataloader(test_dataset, batch_size=cfg["batch_size"])
    
    # MODEL & OTHER OBJECTS
    net = get_model(**cfg["model_params"], device=get_device())
    optimizer = get_optimizer(net, cfg["learning_rate"], cfg["optimizer"])
    loss_fn = get_loss('')
    
    # TRAINING
    net, metrics = train(cfg["nr_epochs"], net, train_dataloader, optimizer, loss_fn, valid_dl=valid_dataloader, greedy_save=True)

    # SAVING
    _path = 'tests'
    if kwargs["_save"]:
        _path = save_model(net, cfg)
        save_training(metrics, _path)
    

    # load best model for test
    try: net = load_model(os.path.join(_path, 'model.pth'), from_config=cfg["model_params"])
    except FileNotFoundError:
        try: net = load_model('checkpoint.pth', from_config=cfg["model_params"])
        except FileNotFoundError: pass

    # GREEDY TESTING
    test_metrics = test(net, test_dataloader)
    if kwargs["_save_stats"]: add_results(cfg, test_metrics)

    # BEAM SEARCH TESTING
    if kwargs.get('test_beam_search', False):
        beam_width = [2, 4, 6, 8]
        test_metrics = {}
        for b in beam_width:
            test_metrics[b] = test_beam_search(net, test_dataloader, beam_width=b)

    return test_metrics


def produce_configurations(params):
    tmp = params.copy()
    param_names = list(tmp.keys())
    if 'model_params' in tmp: tmp["model_params"] = list(produce_configurations(tmp["model_params"]))
    configurations = product(*list(tmp.values()))

    for cfg in configurations:
        yield {k: v for k, v in zip(param_names, cfg)}
        
