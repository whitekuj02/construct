from typing import Any, Dict
import os
import random
import yaml
import argparse
import glob

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from .dataset import CustomDataset, CategoryDataset

def get_train_data_loaders(config: dict[str, Any]):
    
    train_dataset = CustomDataset(config, mode="train")

    return train_dataset


def get_test_data_loaders(config: dict[str, Any]):

    test_dataset = CustomDataset(config, mode="test")

    return test_dataset

def get_category_train_data_loaders(config: dict[str, Any]):
    
    train_dataset = CategoryDataset(config, mode="train")

    return train_dataset


def get_category_test_data_loaders(config: dict[str, Any]):

    test_dataset = CategoryDataset(config, mode="test")

    return test_dataset





if __name__ == "__main__":
    def get_config(config_folder):
        config = {}

        config_folder = os.path.join(config_folder,'*.yaml')
        
        config_files = glob.glob(config_folder)

        for file in config_files:
            with open(file, 'r') as f:
                config.update(yaml.safe_load(f))
        
        if config['device'] == 'cuda' and not torch.cuda.is_available():
            print('using cpu now...')
            config['device'] = 'cpu'

        return config
    
    config = get_config("/home/aicontest/construct/experiment/configs")
    train_dataloader, val_dataloader = get_train_val_data_loaders(config['data'])
    test_dataloader = get_test_data_loaders(config['data'])