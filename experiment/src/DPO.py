from typing import Any, Dict
import os
import random
import yaml
import argparse
import glob
import pandas as pd
import json
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from datasets import Dataset
from transformers import TrainingArguments, Trainer
from trl import SFTTrainer
from trl import DPOTrainer
from trl import DPOConfig

from src.models.model import get_model
from src.dataset.dataset import CustomDataset, CategoryDataset
from src.utils.logger import WandbLogger
from transformers import DataCollatorForSeq2Seq, EarlyStoppingCallback
from src.rag.retriver import rag
from src.utils.metrics import compute_metrics
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset as HFDataset


def run(config: Dict[str, Any]) -> float:
    # 1. 학습한 모델 불러오기
    # 2. 학습한 모델로 train.csv 추론
    # 3. prompt, chosen, rejected 데이터셋 만들기
    # 4. DPO trainer 로 학습

    # 1. 학습한 SFT 모델 불러오기 
    print("1. 학습한 SFT 모델 불러오기")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dir = config['train']['parameter_save'] + '/' + config['train']['test']['dir']
    # 모델과 토크나이저 로드
    model = AutoModelForCausalLM.from_pretrained(dir).to(device)
    tokenizer = AutoTokenizer.from_pretrained(dir)

    sft_model_dir = config['train']['parameter_save'] + '/' + config['train']['test']['dir']
    print(f"sft_model_dir: {sft_model_dir}")

    print(f"학습한 SFT 모델 dir : {sft_model_dir}")

    if tokenizer.pad_token is None : 
        tokenizer.pad_token = tokenizer.eos_token # 만약 모델의 padding token 이 없으면 eos 토큰으로 처리해주기 

    # 2. 기존 CSV 파일이 존재하면 추론 단계 건너뛰기
    csv_path = "/home/aicontest/construct/experiment/result/dpo_data.csv"
    raw_dict = {}

    if os.path.exists(csv_path):
        print(f"기존 CSV 파일을 사용하여 DPO 데이터셋을 생성합니다: {csv_path}")
        df = pd.read_csv(csv_path)
        raw_dict = {
            "prompt": df["prompt"].tolist(),
            "chosen": df["chosen"].tolist(),
            "rejected": df["rejected"].tolist(),
        }
    else:
        print("2. 학습한 모델로 train.csv 추론, rejected 추출하기")
        dataset = CustomDataset(config['data'], mode='train')
        data_items = dataset
        total_items = len(data_items)
        n_rows = total_items // 5
        print(f"Flatten된 데이터 개수: {total_items}, 원본 행 개수: {n_rows}")

        rejected_data = []
        model.to(device)
        
        for i in tqdm(range(n_rows), desc="Generating Rejected"):
            group_5 = data_items[i * 5: i * 5 + 5]
            chosen_item = random.choice(group_5)

        # for chosen_item in tqdm(data_items, desc="Generating Rejected"):
            question = chosen_item["question"]
            choosen_answer = chosen_item["answer"]

            input_text = f"질문: {question}대답:"

            inputs = tokenizer(
                input_text,
                return_tensors="pt",
                truncation=True,
                max_length=512
            ).to(device)

            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=64,
                    do_sample=False,
                    num_beams=1,
                    repetition_penalty=1.2,
                    #no_repeat_ngram_size=2,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

                model_output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
                rejected_answer = model_output.split("대답: ")[-1].strip()
                # print(rejected_answer)

                rejected_data.append({
                    "prompt": question,
                    "chosen": choosen_answer,
                    "rejected": rejected_answer
                })

            del inputs, output_ids
            torch.cuda.empty_cache()

        raw_dict = {
            "prompt": [rd["prompt"] for rd in rejected_data],
            "chosen": [rd["chosen"] for rd in rejected_data],
            "rejected": [rd["rejected"] for rd in rejected_data],
        } 

        df = pd.DataFrame(raw_dict)
        df.to_csv(csv_path, index=False)
        print(f"새로운 DPO 데이터 CSV 파일 생성 완료: {csv_path}")

    dpo_dataset = HFDataset.from_dict(raw_dict)

    print("4. DPO training")

    dpo_args = DPOConfig(
        **config["train"]["dpo"]
    )

    ref_model = AutoModelForCausalLM.from_pretrained(sft_model_dir, torch_dtype=torch.bfloat16).to(device)

    torch.cuda.empty_cache()  # 불필요한 GPU 메모리 해제
    # DPO 모델을 훈련 모드로 설정
    model.train()

    # LoRA 파라미터만 학습 가능하도록 설정
    for name, param in model.named_parameters():
        if "lora" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    ref_model.eval()  # ref_model은 평가 모드 유지 (학습되지 않음)

    dpo_trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=dpo_args,
        train_dataset=dpo_dataset,
        processing_class=tokenizer
    )

    dpo_trainer.train()
    dpo_trainer.save_model(dpo_args.output_dir)

    print(f"DPO 모델 저장 완료: {dpo_args.output_dir}")
    return 0
