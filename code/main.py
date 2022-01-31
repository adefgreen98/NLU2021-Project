import argparse

from pprint import  pprint
from collections import defaultdict
from training_main import main, produce_configurations
from statistics import set_stats_path

import torch

exp1 = {
    "batch_size": [64],
    "optimizer": ["adamw"],
    "learning_rate": [1e-3],
    "nr_epochs": [30],
    "model_params": {
        "bidirectional": [True, False],
        "embedding_method": ['glove'],
        "unit_name": ["lstm", "gru", "rnn"],
        "hidden_size": [256],
        "num_layers": [2],
        "decoder_input_mode": ['label_embed'],
        "intermediate_dropout": [0.0],
        "internal_dropout": [0.0]
    }
}

exp2 = {
    "batch_size": [64],
    "optimizer": ["adam"],
    "learning_rate": [1e-3],
    "nr_epochs": [30],
    "model_params": {
        "bidirectional": [True],
        "embedding_method": ['glove'],
        "unit_name": ["lstm"],
        "hidden_size": [128, 256],
        "num_layers": [1, 2],
        "decoder_input_mode": ['label_embed'],
        "intermediate_dropout": [0.0],
        "internal_dropout": [0.0]
    }
}

exp3 = {
    "batch_size": [64],
    "optimizer": ["adam"],
    "learning_rate": [1e-3],
    "nr_epochs": [30],
    "model_params": {
        "bidirectional": [True],
        "embedding_method": ['glove'],
        "unit_name": ["lstm"],
        "hidden_size": [256],
        "num_layers": [2],
        "decoder_input_mode": ['word+lab', 'sentence', 'label', 'label_nograd', 'label_embed'],
        "intermediate_dropout": [0.0],
        "internal_dropout": [0.0]
    }
}

exp4 = {
    "batch_size": [64],
    "optimizer": ["adam"],
    "learning_rate": [1e-3],
    "nr_epochs": [30],
    "model_params": {
        'num_layers': [2],
        "bidirectional": [True],
        "embedding_method": ['glove'],
        "hidden_size": [256],
        "decoder_input_mode": ['label_nograd'],
        "attention_mode": ['concat', 'global', 'local'],
        "unit_name": ["lstm"]
    }
}

beam_exp = {
    "batch_size": [64],
    "optimizer": ["adam"],
    "learning_rate": [1e-3],
    "nr_epochs": [30],
    "model_params": {
        'num_layers': [2],
        "bidirectional": [True],
        "embedding_method": ['glove'],
        "hidden_size": [256],
        "decoder_input_mode": ['label_nograd'],
        "attention_mode": ['global'],
        "unit_name": ["lstm"]
    }
}

names = {
    'Models_Bidirectionality': exp1,
    'LSTM_architecture': exp2,
    'Decoder_input': exp3,
    'Attention': exp4,
    'Beam_Search': beam_exp
}

parser = argparse.ArgumentParser()

parser.add_argument('--experiment', '--exp', type=str, choices=list(names.keys()) + ['all'], default='all')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--epochs', type=int, default=30)
parser.add_argument('--iterations', type=int, default=10)
parser.add_argument('--save_models', action='store_true')
parser.add_argument('--no_save_stats', action='store_true')

def adjust_args(args_dict):
    args_dict = vars(args_dict)
    bs = args_dict.pop('batch_size')
    epochs = args_dict.pop('epochs')

    if 'all' == args_dict['experiment']:
        # execute all
        res = {}
        for k in names: 
            res[k] = names[k]
            res[k]['batch_size'] = [bs]
            res[k]['nr_epochs'] = [epochs]
        return res
    else:
        # executes only chosen one
        res = {}
        k = args_dict['experiment']
        res[k] = names[k]
        res[k]['batch_size'] = [bs]
        res[k]['nr_epochs'] = [epochs]
        return res


if __name__ == '__main__':
    beam_metrics = defaultdict(list)
    args = parser.parse_args()
    iterations = args.iterations
    save_models = args.save_models
    no_save_stats = args.no_save_stats
    args = adjust_args(args)
    for n,exp in args.items():
        set_stats_path(n + "_stats")
        _cfgs = list(produce_configurations(exp))
        for j, cfg in enumerate(_cfgs):
            for i in range(iterations):
                print(f"---> Experiment: {n}, Iteration {i + 1}/{iterations}, Configuration {j + 1}/{len(_cfgs)} <---") 
                pprint(cfg)
                res = main(cfg, train_path = "data/atis.train.pkl", test_path = "data/atis.test.pkl", _save=save_models, _save_stats=((n!='Beam_Search') and not no_save_stats), test_beam_search=(n=='Beam_Search'))
                if n == 'Beam_Search':
                    for b in res: beam_metrics[b].append(res[b]['f1'])
                print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
                print()
    print("----------------------------------- BEAM MEANS -----------------------------------")
    for k in beam_metrics: print("Beam ", k, ": ", torch.tensor(beam_metrics[k]).mean().item())

    with open('beam_results.txt', 'wt') as f:
        for k in beam_metrics: f.write("Beam ", k, ": ", torch.tensor(beam_metrics[k]).mean().item())

    