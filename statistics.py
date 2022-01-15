"""
* build plots
"""

import os
import datetime
import pandas
import matplotlib.pyplot as plt
import seaborn as sns

from collections import defaultdict

_statistics_path = 'stat_results'
_images_path = os.path.join(_statistics_path, 'images')
_excluded_attributes = ('valid_ratio', 'epochs')
_allowed_columns = ("date", "non-'O' acc.", "accuracy", "precision", "recall", "f1")

if not os.path.exists(_statistics_path): os.mkdir(_statistics_path)
if not os.path.exists(_images_path): os.mkdir(_images_path)

def get_fname(cfg:dict):
    tmp = {k: cfg[k] for k in cfg if k not in _excluded_attributes}
    s = [str(k) + "=" + str(v) for k,v in tmp.items() if k != 'model_params']
    if 'model_params' in tmp: 
        s += [str(k) + "=" + str(v) for k,v in tmp['model_params'].items()]
    s = ";".join(s)
    return s + ".csv"

def available_files():
    return [name for name in os.listdir(_statistics_path) if os.path.isfile(os.path.join(_statistics_path, name))]


def add_results(cfg:dict, _metrics:dict, _save=True):
    fname = get_fname(cfg)
    results = retrieve_results(cfg)
    _metrics["date"] = datetime.datetime.strftime(datetime.datetime.now(), "%Y_%m_%d_%H_%M_%S")
    results = results.append(_metrics, ignore_index=True)
    if _save: results.to_csv(os.path.join(_statistics_path, fname))
    return results

def retrieve_results(cfg:dict):
    fname = get_fname(cfg)
    try: return pandas.read_csv(os.path.join(_statistics_path, fname), index_col=0)
    except FileNotFoundError: return pandas.DataFrame(columns=_allowed_columns) 

def average_single_cfg_metrics(cfg:dict):
    return retrieve_results(cfg).mean(axis=0)


def attrs_from_fname(fname):
    # -4 --> excluding ".csv"
    return {s_attr.split("=")[0]: s_attr.split("=")[1] for s_attr in fname[:-4].split(";")}


def get_attr_possibilities():
    cfgs = [attrs_from_fname(name) for name in available_files()]
    
    # Populating dict of single attribute values
    attr_names = []
    for _cfg in cfgs: attr_names.extend(list(_cfg.keys()))
    attr_names = set(attr_names)
    attr_possibilities = {}
    for k in attr_names:
        attr_possibilities[k] = set()
        for _cfg in cfgs: 
            if k in _cfg: attr_possibilities[k].add(_cfg[k])
    
    return attr_possibilities


def get_single_attr_configs(attribute:str):
    attr_values = get_attr_possibilities()[attribute]
    res_cfgs = defaultdict(list)
    for name in available_files():
        # adds name to list of configs containing specific value
        _cfg = attrs_from_fname(name)
        res_cfgs[_cfg.get(attribute, 'not_found')].append(name)
    return res_cfgs


def dframes_for_attribute(attribute:str):
    # dict attribute value --> filenames containing it
    values_filenames_dict = get_single_attr_configs(attribute) 

    # dict attribute value --> actual DataFrame combination of all configs containing it
    values_dframes_dict = defaultdict(lambda : pandas.DataFrame(columns=_allowed_columns)) 

    # building overall dataframes per each value
    for k in values_filenames_dict:
        values_dframes_dict[k] = values_dframes_dict[k].append([retrieve_results(attrs_from_fname(cfg_name)) for cfg_name in values_filenames_dict[k]])
        values_dframes_dict[k].pop('date')
        values_dframes_dict[k][attribute] = k
    return pandas.DataFrame().append(list(values_dframes_dict.values()))

def build_histogram(*args, notfound_option: dict=None, _savefig=False):
    for attribute in args:
        assert type(attribute) == str
        tmp = dframes_for_attribute(attribute)

        # Building DataFrame with metric values on each line
        l = [[{'metric': k, 'value': el[k], attribute: el[notfound_option[attribute]] if el[attribute] == 'not_found' else el[attribute]} for k in el if k != attribute] for el in tmp.to_dict('records')]
        l = [d for el in l for d in el]
        df = pandas.DataFrame.from_records(l)
        
        # Plotting
        plt.figure(figsize=(10, 6))
        ax = sns.barplot(data=df, hue=attribute, x='metric', y='value')
        for container in ax.containers: ax.bar_label(container, fmt="%.2f", padding=20)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        plt.suptitle(attribute)
        plt.legend(loc='upper left', borderaxespad=-1, bbox_to_anchor=(1.02, 1.0))


        if _savefig: plt.savefig(os.path.join(_images_path, datetime.datetime.strftime(datetime.datetime.now(), "%Y_%m_%d_%H_%M_%S") + f"_{attribute}"))
    plt.show()


if __name__ == '__main__':
    build_histogram(*os.sys.argv[1:], _savefig=False)