from typing import Any, Dict
import os
import random
import yaml
import argparse
import glob

# library
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from datasets import Dataset
from transformers import TrainingArguments, Trainer
from trl import SFTTrainer

# source code
from src.models.model import get_model
# from src.dataset.dataloader import get_train_data_loaders, get_test_data_loaders
from src.dataset.dataset import CustomDataset, CategoryDataset
from src.utils.logger import WandbLogger
from transformers import DataCollatorForSeq2Seq, EarlyStoppingCallback
from src.rag.retriver import rag
from src.utils.metrics import compute_metrics

def run(config: Dict[str, Any]) -> float:

    tokenizer, model = get_model(config['train']['model'])

    if config['data']['category']:
        train_dataset = CategoryDataset(config['data'], mode="train").to_huggingface_dataset()
    else:
        train_dataset = CustomDataset(config['data'], mode="train").to_huggingface_dataset()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    def category_preprocess_function(examples):
        system_prompt = """다음은 사건의 특징 별로 추출된 대표 답변 데이터입니다:
{0}
위의 대표 답변들을 참고하여, 아래 질문에 대해 적절한 답변을 생성해 주세요. 답변 생성 시, 대표 답변이 질문의 내용과 관련이 있다면 반드시 반영해 주시기 바랍니다.
{1}
{2}"""
        instructions = [f"질문: {q}" for q in examples["question"]]
        retrivers =  [f"{q}" for q in examples["retriver"]]
        responses = [f"대답: {a}" for a in examples["answer"]]
        full_texts = [system_prompt.format(retr, inst, resp) for retr, inst, resp in zip(retrivers, instructions, responses)]

        #print(full_texts[0])
        model_inputs = tokenizer(full_texts, padding="max_length", truncation=True, max_length=512)

        input_ids = model_inputs["input_ids"] 
        labels = [ids.copy() for ids in input_ids] 

        for i in range(len(labels)):
            except_answer = full_texts[i].split("대답: ")[0]
            question_ids = tokenizer(f"{except_answer}대답: ", add_special_tokens=False)["input_ids"]
            question_len = len(question_ids)

            labels[i][:question_len] = [-100] * min(question_len, len(labels[i]))

            labels[i] = [-100 if token == tokenizer.pad_token_id else token for token in labels[i]]

        model_inputs["labels"] = labels
        return model_inputs
    
    def preprocess_function(examples):
        instructions = [f"질문: {q} " for q in examples["question"]]
        responses = [f"대답: {a}" for a in examples["answer"]]
        full_texts = [inst + resp for inst, resp in zip(instructions, responses)]

        model_inputs = tokenizer(full_texts, padding="max_length", truncation=True, max_length=512)

        input_ids = model_inputs["input_ids"] 
        labels = [ids.copy() for ids in input_ids] 

        for i in range(len(labels)):
            except_answer = full_texts[i].split("대답: ")[0]
            question_ids = tokenizer(f"{except_answer}대답: ", add_special_tokens=False)["input_ids"]
            question_len = len(question_ids)

            labels[i][:question_len] = [-100] * min(question_len, len(labels[i]))

            # labels[i] = [-100 if token == tokenizer.pad_token_id else token for token in labels[i]]

            eos_idx = [idx for idx, token in enumerate(labels[i]) if token == tokenizer.pad_token_id]

            last_pad_idx = min(eos_idx) if eos_idx else 0 

            labels[i] = [
                -100 if (idx > last_pad_idx and token == tokenizer.pad_token_id) else token
                for idx, token in enumerate(labels[i])
            ]

        model_inputs["labels"] = labels
        return model_inputs
    
    if config['data']['category']:
        train_dataset = train_dataset.map(category_preprocess_function, batched=True)
    else:
        train_dataset = train_dataset.map(preprocess_function, batched=True)
    print(train_dataset[0])

    training_args = TrainingArguments(
        **config['train']['training']
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True, max_length=512)

    if config['train']['SFT']:
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            compute_metrics=compute_metrics,
            tokenizer=tokenizer,
            data_collator=data_collator
        )
    else:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            compute_metrics=compute_metrics,
            tokenizer=tokenizer,
            data_collator=data_collator
        )

    trainer.train()

    dir = config['train']['parameter_save'] + '/' + config['train']['test']['dir']
    os.makedirs(dir, exist_ok=True)
    trainer.save_model(dir)
    tokenizer.save_pretrained(dir)



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
    run(config)