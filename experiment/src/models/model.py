from typing import Any, Dict
import os
import random
import yaml
import argparse
import glob

import torch
import torch.nn as nn

from .base_model import lora_LLM

def get_model(model_config: Dict[str, Any]) -> nn.Module:
    return lora_LLM(model_config).get_model()


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
    model = get_model(config['train']['model'])