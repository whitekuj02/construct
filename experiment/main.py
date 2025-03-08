import os
import random
import yaml
import argparse
import glob

# library
import torch
import numpy as np
import sklearn

# source code
from src import train, test, DPO


def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parse configuration files from a folder')
    
    parser.add_argument('-m', '--mode', help="Select mode(train/test/DPO)", default="train")
    parser.add_argument('-cf', '--config-folder', required=False, help="Path to config folder containing YAML files", default="./configs/")
    parser.add_argument('-sd', '--seed', required=False ,help="whether fixing seed", action='store_false')
    
    args = parser.parse_args()

    config_folder = args.config_folder
    mode = args.mode

    config = get_config(config_folder)

    # seed 고정
    if args.seed:
        set_random_seed(config['random_seed'])

    if mode == 'train':
        train.run(config)
    elif mode == 'test':
        test.run(config)
    elif mode == 'DPO':
        DPO.run(config)
    else:
        raise ValueError(f"Invalid mode: {mode}")