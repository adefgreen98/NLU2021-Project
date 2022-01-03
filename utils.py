import gc

import torch
import torch.nn as nn
from model import Seq2SeqModel
from dataset import *

def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

"""# Getters"""

"""Dataset"""
def get_dataset(path, embedder, name='atis'):
    """
    Prepares the dataset with the specified name. Currently supported:
    - 'atis': ATIS dataset 
    """
    if name == 'atis': return ATISDataset(path, embedder)


"""Dataloader"""
def get_dataloader(dataset, collate_type=None, batch_size=32, num_workers=2, shuffle=False):
    _collate_fn = None
    if collate_type == 'ce':
        _collate_fn = dataset.collate_ce
    return torch.utils.data.DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, collate_fn=_collate_fn, shuffle=shuffle)


"""Model"""
def get_model(model_type, labels, input_size, hidden_size=256, device='cuda'):
    return Seq2SeqModel(model_type, labels, input_size, hidden_size=hidden_size, device=device).to(device)


"""Optimizer"""
def get_optimizer(model, lr=0.001, name="adam"):
    if name == 'adam': return torch.optim.Adam(model.parameters(), lr)
    elif name == 'sgd': return torch.optim.SGD(model.parameters(), lr)
    elif name == 'adamw': return torch.optim.AdamW(model.parameters, lr)
    else: raise RuntimeError("unexpected optimizer name " + str(name))


"""Loss function"""
def get_loss(name):
    if name == 'cross_entropy': return nn.CrossEntropyLoss()

    
"""Accuracy"""
def compute_accuracy(yp, yt, lab2idx):
    confusion_matrix = torch.zeros(len(lab2idx), len(lab2idx), dtype=torch.int)
    for i in range(len(yt)): 
        for idx in zip(yt[i], yp[i]):
            confusion_matrix[idx] = confusion_matrix[idx] + 1
    d = {lab: confusion_matrix[lab2idx[lab], lab2idx[lab]].item() / confusion_matrix[lab2idx[lab]].sum()  for lab in lab2idx}
    d["total"] = sum(list(d.values())) / len(d)
    return d


"""Templates for train, validate & test:"""
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
    global_acc = compute_accuracy(*acc_history, model.lab2idx) #here using list of all outputs from each batch

    del loss_history, acc_history
    gc.collect()
    
    #TODO: tensorboard or something to plot live
    return mean_loss, global_acc["total"]


def train(nr, model, train_dl, optimizer, loss_fn, valid_dl=None):
    
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
            if metrics['eval']["mean_acc"][e] > best_acc:
                best_acc = metrics['eval']["mean_acc"][e]
                torch.save(model, 'checkpoint.pth')
        else: 
            # Saving model as better metrics in train
            if metrics['train']["mean_acc"][e] > best_acc:
                best_acc = metrics['train']["mean_acc"][e]
                torch.save(model, 'checkpoint.pth')
                
        print("")
    
    return metrics

"""# Plots"""