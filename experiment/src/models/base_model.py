from typing import Any, Dict
import os
import random
import yaml
import argparse
import glob

import torch
import torch.nn as nn
import bitsandbytes as bnb
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, TaskType, get_peft_model

class lora_LLM:
    def __init__(self, config: Dict[str, Any]):

        self.llm_name = config['name']
        self.use_quantization = config['quantization']['use']
        self.quantization_config = config['quantization']['config']

        self.tokenizer = AutoTokenizer.from_pretrained(self.llm_name)
        if self.use_quantization:
            self.quantization_config['bnb_4bit_compute_dtype'] = getattr(torch, self.quantization_config['bnb_4bit_compute_dtype'])
            self.bnb_config = BitsAndBytesConfig(
                **self.quantization_config
            )

            self.llm = AutoModelForCausalLM.from_pretrained(self.llm_name, quantization_config=self.bnb_config, device_map="auto")
        else:
            self.llm = AutoModelForCausalLM.from_pretrained(self.llm_name, device_map="auto")
        
        for name, param in self.llm.named_parameters():
            param.requires_grad = False

        self.use_lora = config['lora']['use']
        self.lora_config = config['lora']['config']

        if self.use_lora:
            self.peft_config = LoraConfig(
                **self.lora_config
            )
            self.model = get_peft_model(self.llm, self.peft_config)
        
        self.model.print_trainable_parameters()
    
    def get_model(self):
        return self.tokenizer, self.model


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
    _model = lora_LLM(config['train']['model'])