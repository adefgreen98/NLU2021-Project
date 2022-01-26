"""
* build plots
"""

import os
import datetime
import pandas
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from collections import defaultdict
from utils import stoval

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
    return {s_attr.split("=")[0]: stoval(s_attr.split("=")[1]) for s_attr in fname[:-4].split(";")}


def get_attr_possibilities():
    """
    Returns a dictionary associating each attribute to its possible values.
    """
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


def get_single_attr_configs(attribute:str, to_select_attrs=None):
    """
    Associates each value of the attribute to all the configurations containing it. Allows to specify which 
    configurations to exclude through an additional dictionary mapping (other attribute names, [allowed values list]).
    """
    if to_select_attrs is not None:
        if len(to_select_attrs) > 0 and type(to_select_attrs[list(to_select_attrs.keys())[0]]) != list:
            raise ValueError("this function is supposed to be used for sets of values; other usages could cause unpredicted behavior")

    attr_values = get_attr_possibilities()[attribute]
    res_cfgs = defaultdict(list)
    for name in available_files():
        # adds name to list of configs containing specific value
        _cfg = attrs_from_fname(name)
        
        # print(_cfg)
        # print({k: (_cfg.get(k, 'not_found'), _cfg.get(k, 'not_found') not in to_select_attrs[attribute][k]) for k in to_select_attrs[attribute]})
        # os.system("pause")
        
        if to_select_attrs is not None:
            if any([_cfg.get(k, 'not_found') not in to_select_attrs[k] for k in to_select_attrs]):
                # if any of the values contained in current config is not contained in the set of values specified for 
                # other attributes by the current attribute ...
                continue # ... skipping the current name because attributes to select do not match
        res_cfgs[_cfg.get(attribute, 'not_found')].append(name)
    return res_cfgs


def dframes_for_attribute(attribute:str, to_select_attrs=None):
    """
    Returns a DataFrame containing all training results for one specific attribute, with one
    additional column indicating the attribute's value.
    """
    # dict attribute value --> filenames containing it
    values_filenames_dict = get_single_attr_configs(attribute, to_select_attrs) 

    # dict attribute value --> actual DataFrame combination of all configs containing it
    values_dframes_dict = defaultdict(lambda : pandas.DataFrame(columns=_allowed_columns)) 

    # building overall dataframes per each value
    for k in values_filenames_dict:
        values_dframes_dict[k] = values_dframes_dict[k].append([retrieve_results(attrs_from_fname(cfg_name)) for cfg_name in values_filenames_dict[k]])
        values_dframes_dict[k].pop('date')
        values_dframes_dict[k][attribute] = k
    return pandas.DataFrame().append(list(values_dframes_dict.values()))


def retrieve_metric_dataframe(attribute, to_select_attrs=None, notfound_option:dict=None):
    """
    Creates a DataFrame that associates (attribute value, metric name, metric value). Allows also to select a subset of
    trainings by specifying other attributes' values. 
    """
    
    retr_df = dframes_for_attribute(attribute, to_select_attrs)
    l = []
    for el in retr_df.to_dict('records'):
        for k in el:
            if k != attribute:
                mdict = {'metric': k, 'value': el[k]}
                mdict[attribute] = el[attribute]
                if notfound_option is not None: 
                    if el[attribute] == 'not_found':
                        mdict[attribute] = notfound_option.get(attribute, 'not_found')
                l.append(mdict) 

    return pandas.DataFrame.from_records(l)



def build_histogram(*args, to_select_attrs:dict={}, notfound_option:dict=None, _savefig=False):

    for attribute in args:
        assert type(attribute) == str

        df = retrieve_metric_dataframe(attribute, to_select_attrs.get(attribute, None), notfound_option)

        # Plotting
        plt.figure(figsize=(10, 10))
        attr_metrics = set(df['metric'].tolist())
            
        # attr_values = order_values(set(df[attribute].tolist()))
        attr_values = sorted(set(df[attribute].tolist()))

        colors = sns.color_palette('pastel', n_colors=len(attr_values))
        _step = 4
        _separation = _step * (len(attr_values) + 1)
        _reduction = .8

        # Error bar utilities
        _ci = .8
        _use_errs = False
        
        attr_pos = [list(np.arange(m - (len(attr_values) // 2)*_step , m + (len(attr_values) // 2)*_step + 1, _step)) for m in range(0, _separation*len(attr_metrics), _separation)]
        for i, m in enumerate(attr_metrics):
            for j, k in enumerate(attr_values):
                val_mean = df[(df[attribute] == k) & (df['metric'] == m)]['value'].mean()
                val_std = (_ci * df[(df[attribute] == k) & (df['metric'] == m)]['value'].std()) if _use_errs else 0 
                container = plt.barh(attr_pos[i][j], val_mean, height=_reduction*_step, label=k, color=colors[j], xerr=val_std)
                plt.bar_label(container, fmt="%.2f", padding=int(val_std) + 3)

        plt.legend(attr_values, loc='upper left', bbox_to_anchor=(.9, 1.0))
        plt.title(attribute)
        plt.xlabel("Value(%)")
        plt.ylabel("Metric")
        plt.xlim((df['value'].min() - 5, 100))
        plt.yticks(ticks=[pos[len(pos) // 2] for pos in attr_pos], labels=attr_metrics)
        sns.despine()

        # plt.figure(figsize=(10, 6))
        # ax = sns.barplot(data=df, hue=attribute, y='metric', x='value', orient='h')
        # ax.set(xlim=[60, 105])
        # for container in ax.containers: 
        #     ax.bar_label(container, fmt="%.2f", padding=25)
        # ax.spines['right'].set_visible(False)
        # ax.spines['top'].set_visible(False)
        # plt.suptitle(attribute)
        # plt.legend(loc='upper left', borderaxespad=-1, bbox_to_anchor=(1.02, 1.0))


        if _savefig: plt.savefig(os.path.join(_images_path, datetime.datetime.strftime(datetime.datetime.now(), "%Y_%m_%d_%H_%M_%S") + f"_{attribute}"))
    plt.show()

def compute_diff(attribute, metric='f1'):
    """
    Computes mean difference between equal conditions except for the single parameter specified
    """



if __name__ == '__main__':

    if len(os.sys.argv[1:]) != 0: build_histogram(*os.sys.argv[1:], _savefig=False)
    else:
        attrs = ['internal_dropout', 'intermediate_dropout', 'num_layers']

        to_choose_all = {'num_layers': [2], 'bidirectional': [True], 'batch_size': [64]}
        to_choose_layers = {'bidirectional': [True], 'batch_size': [64]}

        build_histogram(*attrs,
            to_select_attrs={
                'internal_dropout': to_choose_all, 
                'intermediate_dropout': to_choose_all, 
                'num_layers': to_choose_layers
            },
            _savefig=False
            )