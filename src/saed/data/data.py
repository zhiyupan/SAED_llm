import os
import os.path as osp
import numpy as np
import pandas as pd
from omegaconf import DictConfig
here = osp.dirname(osp.abspath(__file__))
root_path = osp.join(here, '..', '..', '..')

def load_table_list(config: DictConfig) -> pd.DataFrame:
    data = pd.read_csv(osp.join(root_path, config["data"]["tables"]["path"], 'table_list.csv'))
    return data

def load_tables(config: DictConfig) -> dict:
    table_list = load_table_list(config=config)
    tables = {}
    for table_id in table_list['table_id']:
        tables[table_id] = pd.read_csv(osp.join(root_path, config["data"]["tables"]["path"], table_id))
    return tables

def load_labels(config: DictConfig) -> pd.DataFrame:
    # data_small = pd.read_csv(osp.join(root_path, config["data"]["labels"]["path"],'ground_truth_small_table_based_on_name.csv'))
    # data_large = pd.read_csv(osp.join(root_path, config["data"]["labels"]["path"], 'ground_truth_large_table_based_on_name.csv'))
    # return pd.concat([data_small, data_large])
    return pd.read_csv(osp.join(root_path, config["data"]["labels"]["path"], 'ground_truth.csv'))


def load_table(table_id:str, config: DictConfig) -> pd.DataFrame:
    data = pd.read_csv(osp.join(root_path, config["data"]["tables"]["path"], table_id))
    return data

