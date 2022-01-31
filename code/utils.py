import os
import shutil
import gc
import tqdm

import pandas as pd
import torch
import torch.nn as nn

from random import shuffle

from model import Seq2SeqModel
from attention import AttentionSeq2SeqModel

from dataset import ATISDataset

from loss import *
from conlleval import evaluate as conll_evaluate

def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

def stoval(s):
    if s == "True": return True 
    elif s == "False": return False

    try: return int(s)
    except ValueError:
        try: return float(s)
        except ValueError:
            return s

"""# Getters"""

"""Dataset"""
def get_dataset(path, name='atis'):
    """
    Prepares the dataset with the specified name. Currently supported:
    - 'atis': ATIS dataset 
    """
    if name == 'atis': return ATISDataset(path)

def split_dataset(dataset, valid_ratio=0.1, rnd=False):
    """
    Involved in splitting training set and validation set from the original dataset.
    Inputs:
    :param ratio: percentage of samples to put in validation set
    Outputs:
    :param training_set: torch.utils.data.Subset containing samples for training
    :param valid_set: same as training_set, but with samples for evaluation
    """
    _idx = list(range(len(dataset)))
    _last = round(len(_idx) * (1 - valid_ratio))
    if rnd: shuffle(_idx)
    train = torch.utils.data.Subset(dataset, _idx[:_last])
    valid = torch.utils.data.Subset(dataset, _idx[_last:])
    return train, valid


"""Dataloader"""
def get_dataloader(dataset, batch_size=32, num_workers=0, shuffle=False):
    def collate(batch): return [b[0] for b in batch], [b[1] for b in batch]
    return torch.utils.data.DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, collate_fn=collate, shuffle=shuffle)


"""Model"""
def get_model(**model_params):
    if 'attention_mode' not in model_params or model_params['attention_mode'] == 'none':
        return Seq2SeqModel(**model_params).to(get_device())
    else: return AttentionSeq2SeqModel(**model_params)


def load_model(path, from_config:dict=None):

    if from_config is None:
        model_params = os.path.split(path)[1] #counting relative path from the topmost 'models' folder
        model_params = {s_attr.split("=")[0]: stoval(s_attr.split("=")[1]) for s_attr in model_params.split(";")}
        model = get_model(**model_params, device=get_device())
    else:
        model_params = from_config
        model = get_model(**model_params, device=get_device())
    model.load_state_dict(torch.load(path))
    return model


"""Optimizer"""
def get_optimizer(model, lr=0.001, name="adam"):
    if name == 'adam': return torch.optim.Adam(model.parameters(), lr)
    elif name == 'sgd': return torch.optim.SGD(model.parameters(), lr)
    elif name == 'adamw': return torch.optim.AdamW(model.parameters(), lr)
    else: raise RuntimeError("unexpected optimizer name " + str(name))


def get_loss(name):
    return torch.nn.CrossEntropyLoss()

    
"""# Metrics"""
def custom_compute_accuracy(yp, yt, lab2idx, weigh_classes=True):
    confusion_matrix = torch.zeros(len(lab2idx), len(lab2idx), dtype=torch.int)
    for i in range(len(yt)): 
        for idx in zip(yt[i], yp[i]):
            confusion_matrix[idx] = confusion_matrix[idx] + 1

    # d = {lab: confusion_matrix[lab2idx[lab], lab2idx[lab]].item() / confusion_matrix[lab2idx[lab]].sum()  for lab in lab2idx}
    
    ## Needed for debug ##
    idx2lab = {v:k for k,v in lab2idx.items()}
    tmp = {idx2lab[i]: (v[0], v[1]) for i, v in enumerate(list(zip(confusion_matrix.diag().tolist(), confusion_matrix.sum(dim=-1).tolist())))}
    ## ##

    if weigh_classes:
        accuracies = (confusion_matrix.diag() / confusion_matrix.sum(dim=-1)) * (confusion_matrix.sum(dim=-1) / confusion_matrix.sum())
        d = {lab: accuracies[lab2idx[lab]].item() if not accuracies[lab2idx[lab]].isnan().item() else -1.0 for lab in lab2idx} #excluding NaN results and setting them to -1.0
        d["total"] = sum([v for v in d.values() if v >= 0]) # mean is obtain thanks to previous weighting
    else:
        accuracies = (confusion_matrix.diag() / confusion_matrix.sum(dim=-1))
        d = {lab: accuracies[lab2idx[lab]].item() if not accuracies[lab2idx[lab]].isnan().item() else -1.0 for lab in lab2idx} #excluding NaN results and setting them to -1.0
        d["total"] = sum([v for v in d.values() if v >= 0]) / len([v for v in d.values() if v >= 0]) #computing mean excluding negative values

    return d


def compute_accuracy(yp, yt):
    res = conll_evaluate(yt, yp, verbose=False)
    acc, f1 = res[1], res[-1]
    return acc, f1



"""# Templates for train, validate & test:"""
def iterate(model, dataloader, optimizer, loss_fn, mode):
    if mode == 'train': model.train()
    else: model.eval()

    loss_history = torch.zeros(len(dataloader), dtype=torch.float)
    acc_history = {'predictions': [], 'trues': []}

    progbar_length = len(dataloader)
    progbar = tqdm.tqdm(position=0, leave=False, ncols=70)
    progbar.reset(total=progbar_length)

    if mode == 'train': 
        for i, data in enumerate(dataloader):
            optimizer.zero_grad()
            loss, yp, yt = model.process_batch(data, loss_fn)
            loss.backward()
            optimizer.step()

            loss_history[i] = loss

            acc_history['predictions'].append(yp)
            acc_history['trues'].append(yt)

            progbar.update()
    else:
        with torch.no_grad():
            for i, data in enumerate(dataloader):
                loss, yp, yt = model.process_batch(data, loss_fn)
                loss_history[i] = loss

                acc_history['predictions'].append(yp)
                acc_history['trues'].append(yt)
                
                progbar.update()

    return loss_history, acc_history


def epoch(model, dataloader, optimizer, loss_fn, mode, test_set=None):

    loss_history, acc_history = iterate(model, dataloader, optimizer, loss_fn, mode)
    mean_loss = loss_history.mean()
    yp, yt = acc_history['predictions'], acc_history['trues']

    # concatenating batches
    ep_yp, ep_yt = [model.idx2tag[i] for el in yp for i in el], [model.idx2tag[i] for el in yt for i in el]

    # global_acc = custom_compute_accuracy(*acc_history, model.embedder.tag2idx)['total']
    acc, f1 = compute_accuracy(ep_yp, ep_yt) 

    return mean_loss, acc, f1


def train(nr, model, train_dl, optimizer, loss_fn, valid_dl=None, greedy_save=False, test_set=None):
    
    metrics = {}
    metrics["train"] = {
        "mean_loss": torch.zeros(nr, dtype=torch.float), 
        "mean_acc": torch.zeros(nr, dtype=torch.float),
        "mean_f1": torch.zeros(nr, dtype=torch.float)
        }
    if valid_dl is not None:
        metrics["eval"] = {
            "mean_loss": torch.zeros(nr, dtype=torch.float), 
            "mean_acc": torch.zeros(nr, dtype=torch.float),
            "mean_f1": torch.zeros(nr, dtype=torch.float)
            }

    best_acc = -1
    for e in range(nr):
        print(f"------------- EPOCH {e + 1}/{nr} -------------")

        # TRAIN
        metrics['train']["mean_loss"][e], metrics['train']["mean_acc"][e], metrics['train']["mean_f1"][e] = epoch(model, train_dl, optimizer, loss_fn, 'train')
        print(f"""train : loss = {metrics['train']["mean_loss"][e].item() : .4f} | accuracy = {metrics['train']["mean_acc"][e].item() : .4f} | f1 = {metrics['train']["mean_f1"][e] : .4f}""")
        
        # EVAL
        if valid_dl is not None:
            metrics['eval']["mean_loss"][e], metrics['eval']["mean_acc"][e], metrics['eval']["mean_f1"][e] = epoch(model, valid_dl, optimizer, loss_fn, 'eval', test_set=test_set)
            print(f"""{'eval'} : loss = {metrics['eval']["mean_loss"][e].item() : .4f} | accuracy = {metrics['eval']["mean_acc"][e].item() : .4f} | f1 = {metrics['eval']["mean_f1"][e] : .4f}""")

            # Saving model as better evaluation
            if greedy_save:
                if metrics['eval']["mean_acc"][e] > best_acc:
                    best_acc = metrics['eval']["mean_acc"][e]
                    torch.save(model.state_dict(), 'checkpoint.pth')
        else: 
            # Saving model as better metrics in train
            if greedy_save:
                if metrics['train']["mean_acc"][e] > best_acc:
                    best_acc = metrics['train']["mean_acc"][e]
                    torch.save(model.state_dict(), 'checkpoint.pth')
                
        print("")
    
    return model, metrics



def test(model, test_dataloader):

    model.eval() 

    _, acc_history = iterate(model, test_dataloader, None, lambda a,b:torch.tensor(0.0), 'eval')
    
    # concatenating batches
    ep_yp, ep_yt = [model.idx2tag[i] for el in acc_history['predictions'] for i in el], [model.idx2tag[i] for el in acc_history['trues'] for i in el]
    
    print("")
    result = conll_evaluate(ep_yp, ep_yt, verbose=False)
    print("")
    print("------------------ TEST RESULT ------------------")
    print("accuracy (non-'O') = {:.4f} | accuracy = {:.4f} | precision = {:.4f}  |  recall = {:.4f}  |  f1 = {:.4f}".format(*result))
    return {"non-'O' acc.": result[0], "accuracy": result[1], "precision": result[2], "recall": result[3], "f1": result[4]}


# BEAM SEARCH UTILITIES
metric_idx = {"non-'O' acc.": 0, "accuracy": 1, "precision": 2, "recall": 3, "f1": 4}



def test_beam_search(net, dataloader, nr_sentences=None, beam_width=5, save_path='tests', fname=None, rnd=True):
    
    net.eval()

    if fname is None:  fname = f"beam_{beam_width}.txt"
    fname = os.path.join(save_path, fname)

    toprint = []
    true_seqs = []
    pred_seqs = []

    print(f"------------------ BEAM SEARCH {beam_width} TEST RESULT ------------------")

    progbar_length = len(dataloader)
    progbar = tqdm.tqdm(position=0, leave=False, ncols=70)
    progbar.reset(total=progbar_length)

    for data in dataloader:
        _str = "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n"
        s, beam, score, yt = net.beam_inference(data, beam_width=beam_width) 

        # data is in batched mode
        for batch_idx in range(len(s)):
            true_seqs.extend(yt[batch_idx])
            pred_seqs.extend(beam[batch_idx][0]) # best beam prediction is first
            _str += f"{'Sentence' : <15}{'Label' : ^30}" + "".join([f"{'Score ' + str(score[batch_idx][i]) : ^30}" for i in range(len(score[batch_idx]))]) + "\n"
            _str += "---------------------------------------------------------------------------------\n"
            for i in range(len(s[batch_idx])):
                # beam[batch_idx][j][i] because printing column-wise
                _str += f"{s[batch_idx][i] : <15}{yt[batch_idx][i] : ^30}" + "".join([f"{beam[batch_idx][j][i] : ^30}" for j in range(len(beam[batch_idx]))]) + "\n" 
            _str += "\n"
            toprint.append(_str)

        progbar.update()
    print('') # separate from progbar

    result = conll_evaluate(true_seqs, pred_seqs, verbose=False)
    print("accuracy (non-'O') = {:.4f} | accuracy = {:.4f} | precision = {:.4f}  |  recall = {:.4f}  |  f1 = {:.4f}".format(*result))
    with open(fname, 'wt') as f:
        for i, s in enumerate(toprint): 
            f.write(s)
            f.write("\n\n")
    return {"non-'O' acc.": result[0], "accuracy": result[1], "precision": result[2], "recall": result[3], "f1": result[4]}

"""# Plots"""


"""# Saves"""

def save_model(model, configuration:dict, models_folder='models'):
    if not os.path.exists(models_folder): os.mkdir(models_folder)

    dirname = ";".join([str(k) + "=" + str(v) for k,v in configuration.items() if k != 'model_params'] + [str(k) + "=" + str(v) for k,v in configuration['model_params'].items()])
    dirname = os.path.join("models", dirname)
    if os.path.exists(dirname): shutil.rmtree(dirname)
    os.mkdir(dirname)

    # automatically detects if there is a better checkpoint already
    model_path = os.path.join(dirname, "model.pth")
    if os.path.exists('./checkpoint.pth'): shutil.move('./checkpoint.pth', model_path)
    else: torch.save(model.state_dict(), model_path)
    
    return dirname


def save_training(metrics:dict, path):
    """
    Saves training metrics history as a .csv file (so that it can be easily graphed).
    The .csv will contain one row for each epoch, and columns will represent different metrics at that point.
    :param metrics: a dictionary associating metric name to the list of values it scored for each epoch
    :param path: path of parent directory where the file will be saved (i.e. experiment directory)
    """
    #TODO: insert training / validation metrics
    for mode in ['train', 'eval']:
        mode_metric = {k: v.detach().numpy() for k,v in metrics[mode].items()}
        pd.DataFrame.from_dict(mode_metric).to_csv(os.path.join(path, f"{mode}.csv"))