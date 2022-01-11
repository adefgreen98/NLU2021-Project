import os
import shutil
import gc

import pandas as pd
import torch
import torch.nn as nn

from model import Seq2SeqModel

from dataset import ATISDataset

from loss import *
from embedder import Embedder
from conlleval import evaluate as conll_evaluate

def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

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
def get_dataloader(dataset, collate_type:str=None, batch_size=32, num_workers=0, shuffle=False):
    _collate_fn = None
    if collate_type == 'ce':
        _collate_fn = Embedder.collate_ce
    return torch.utils.data.DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, collate_fn=_collate_fn, shuffle=shuffle)


"""Model"""
def get_model(embedder, **model_params):
    return Seq2SeqModel(embedder, **model_params).to(get_device())

def load_model(path):
    raise NotImplementedError("still to understand how to load state dict without access to model information")


"""Optimizer"""
def get_optimizer(model, lr=0.001, name="adam"):
    if name == 'adam': return torch.optim.Adam(model.parameters(), lr)
    elif name == 'sgd': return torch.optim.SGD(model.parameters(), lr)
    elif name == 'adamw': return torch.optim.AdamW(model.parameters(), lr)
    else: raise RuntimeError("unexpected optimizer name " + str(name))


"""Loss function"""
def get_loss(name, embedder):
    """
    Returns the chosen loss. The 2nd argument is needed for MaskedLoss instantiation.
    """
    if name == 'ce': 
        return nn.CrossEntropyLoss().to(get_device())
    elif name == 'masked_ce': 
        return MaskedLoss(nn.CrossEntropyLoss().to(get_device()), embedder.get_padding_token_index()).to(get_device())
    else: 
        raise ValueError(f"loss {str(name)} not supported")

    
"""# Metrics"""
def compute_accuracy(yp, yt, lab2idx, weigh_classes=True):
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


"""# Templates for train, validate & test:"""
def iterate(model, dataloader, optimizer, loss_fn, mode):
    if mode == 'train': model.train()
    else: model.eval()

    loss_history = torch.zeros(len(dataloader), dtype=torch.float)
    acc_history = [[],[]]
    if mode == 'train': 
        for i, data in enumerate(dataloader):
            optimizer.zero_grad()
            loss, acc_labels_list = model.process_batch(data, loss_fn)
            loss.backward()
            optimizer.step()

            loss_history[i] = loss

            #TODO: refine accuracy?
            acc_history[0].append(acc_labels_list[0])
            acc_history[1].append(acc_labels_list[1])
    else:
        with torch.no_grad():
            for i, data in enumerate(dataloader):
                loss, acc_labels_list = model.process_batch(data, loss_fn)
                loss_history[i] = loss

                #TODO: refine accuracy?
                acc_history[0].append(acc_labels_list[0])
                acc_history[1].append(acc_labels_list[1])

    return loss_history, acc_history


def epoch(model, dataloader, optimizer, loss_fn, mode):

    loss_history, acc_history = iterate(model, dataloader, optimizer, loss_fn, mode)
    mean_loss = loss_history.mean()
    global_acc = compute_accuracy(*acc_history, model.embedder.tag2idx) #here using list of all outputs from each batch

    del loss_history, acc_history
    gc.collect()
    
    #TODO: tensorboard or something to plot live
    return mean_loss, global_acc["total"]


def train(nr, model, train_dl, optimizer, loss_fn, valid_dl=None, greedy_save=False):
    
    metrics = {}
    metrics["train"] = {"mean_loss": torch.zeros(nr, dtype=torch.float), "mean_acc": torch.zeros(nr, dtype=torch.float)}
    if valid_dl is not None:
        metrics["eval"] = {"mean_loss": torch.zeros(nr, dtype=torch.float), "mean_acc": torch.zeros(nr, dtype=torch.float)}

    best_acc = -1
    for e in range(nr):
        print(f"------------- EPOCH {e + 1}/{nr} -------------")

        # TRAIN
        metrics['train']["mean_loss"][e], metrics['train']["mean_acc"][e] = epoch(model, train_dl, optimizer, loss_fn, 'train')
        print(f"""train : loss = {metrics['train']["mean_loss"][e].item()} | accuracy = {metrics['train']["mean_acc"][e].item()}""")
        
        # EVAL
        if valid_dl is not None:
            metrics['eval']["mean_loss"][e], metrics['eval']["mean_acc"][e] = epoch(model, valid_dl, optimizer, loss_fn, 'eval')
            print(f"""{'eval'} : loss = {metrics['eval']["mean_loss"][e].item()} | accuracy = {metrics['eval']["mean_acc"][e].item()}""")

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
    
    return metrics




def test(model, dataset, nr_sentences=None, save_path='tests', fname=None, rnd=True):

    model.eval() 

    if nr_sentences is None: nr_sentences = len(dataset)
    
    if rnd: idxs = torch.randperm(len(dataset))[:nr_sentences].tolist()
    else: idxs = range(nr_sentences)

    if fname is None: fname = "greedy.txt"
    fname = os.path.join(save_path, fname)
    
    toprint = []
    true_seqs = []
    pred_seqs = []
    for i in idxs:
        sent, lab = dataset[i]
        yp = model.run_inference(sent)

        s, l = model.embedder.get_test_sentence(sent, lab) #regularize with SOS, EOS for printing & testing
        
        true_seqs.extend(l)
        pred_seqs.extend(yp)

        _str = ""
        for el in zip(s, l, yp):
            _str += f"{el[0] : <15}{el[1] : ^30}{el[2] : ^30}\n"
        toprint.append(_str)
    
    result = conll_evaluate(true_seqs, pred_seqs, verbose=False)
    print("------------------ TEST RESULT ------------------")
    print("accuracy (non-'O') = {:.4f} | accuracy = {:.4f} | precision = {:.4f}  |  recall = {:.4f}  |  f1 = {:.4f}".format(*result))
    print("-------------------------------------------------")
    
    with open(fname, 'wt') as f:
        f.write(" ".join([str(el) for el in result]) + "\n")
        f.write("\n")
        for i, s in enumerate(toprint): 
            # if i < 5: print(s)
            f.write(s)
            f.write("\n\n")


def test_beam_search(net, dataset, nr_sentences=None, beam_width=5, save_path='tests', fname=None, rnd=True):
    
    net.eval()

    if nr_sentences is None: nr_sentences = len(dataset)
    if rnd: idxs = torch.randperm(len(dataset))[:nr_sentences].tolist()
    else: idxs = range(nr_sentences)

    if fname is None:  fname = "beam.txt"
    fname = os.path.join(save_path, fname)

    toprint = []
    for i in idxs:
        _str = "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n"
        sent, lab = dataset[i]
        beam, score = net.beam_inference(sent) 
        
        sent, lab = net.embedder.get_test_sentence(sent, lab) #regularize with SOS, EOS for printing
        _str += f"{'Sentence' : <15}{'Label' : ^30}" + "".join([f"{'Score ' + str(score[i]) : ^30}" for i in range(len(score))]) + "\n"
        _str += "---------------------------------------------------------------------------------\n"
        for i in range(len(sent)):
            _str += f"{sent[i] : <15}{lab[i] : ^30}" + "".join([f"{beam[j][i] : ^30}" for j in range(len(beam))]) + "\n"
        _str += "\n"
        toprint.append(_str)
    
    #TODO: how to compute metrics when we have multiple results? maybe the most correct in the beam?
    
    with open(fname, 'wt') as f:
        for i, s in enumerate(toprint): 
            f.write(s)
            f.write("\n\n")

"""# Plots"""


"""# Saves"""

def save_model(model, configuration:dict):
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