from typing import Any, Dict
import os
import random
import yaml
import argparse
import glob
import re
# library
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from datasets import Dataset
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA, LLMChain
from langchain.llms import HuggingFacePipeline

from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTTrainer
from sentence_transformers import SentenceTransformer
import pandas as pd

# source code
from src.models.model import get_model
from src.dataset.dataloader import get_test_data_loaders
from src.utils.logger import WandbLogger
from transformers import DataCollatorForSeq2Seq
from src.rag.retriver import rag
from src.dataset.dataset import CustomDataset, CategoryDataset
from src.utils.metrics import compute_metrics

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import pandas as pd
from datasets import Dataset
from sentence_transformers import SentenceTransformer
from typing import Dict, Any

def run(config: Dict[str, Any]) -> float:

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dir = config['train']['parameter_save'] + '/' + config['train']['test']['dir']
    dir = "/home/aicontest/construct/experiment/dpo_model"
    print(dir)
    # 모델과 토크나이저 로드
    model = AutoModelForCausalLM.from_pretrained(dir).to(device)
    tokenizer = AutoTokenizer.from_pretrained(dir)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"모델 설정 확인 : {model.config}")
    
    # 테스트 데이터 로드
    if config['data']['category']:
        test_dataset = CategoryDataset(config['data'], mode="test").to_huggingface_dataset()
    else:
        test_dataset = CustomDataset(config['data'], mode="test").to_huggingface_dataset()

    print("테스트 실행 시작... 총 테스트 샘플 수:", len(test_dataset))

    # 결과 저장 리스트
    test_results = []

    # 모델 추론 실행
    for idx, row in enumerate(test_dataset):
        if (idx + 1) % 10 == 0 or idx == 0:
            print(f"\n[샘플 {idx + 1}/{len(test_dataset)}] 진행 중...")
        
        if config['data']['category']:
            question = f"""다음은 사건의 특징 별로 추출된 대표 답변 데이터입니다:
{row['retriver']}
위의 대표 답변들을 참고하여, 아래 질문에 대해 적절한 답변을 생성해 주세요. 답변 생성 시, 대표 답변이 질문의 내용과 관련이 있다면 반드시 반영해 주시기 바랍니다.
질문: {row['question']}
대답: """
        else:
            question = f"질문: {row['question']}대답:"
        #print(question)

        # 모델 생성 실행 (generate 사용)
        input_ids = tokenizer(
            question, 
            truncation=True,  # 길이가 max_length보다 길면 잘라냄
            max_length=512,   # 최대 길이 제한
            return_tensors="pt",  
            padding=False  # 패딩 제거 (또는 생략 가능)
        ).to(device)

        # print(f"input_id : {input_ids}")

        output_ids = model.generate(
            **input_ids,  # 질문을 그대로 입력
            max_new_tokens=128,  # 답변 길이만 제한
            do_sample=True,  # 확률적 샘플링 비활성화 (일관된 결과 보장)
            num_beams=1,  # Greedy Decoding 사용 (최적의 답변 출력)
            repetition_penalty=1.2,
            pad_token_id=tokenizer.pad_token_id,  # 패딩을 EOS로 설정
            eos_token_id=tokenizer.eos_token_id  # EOS에서 멈추도록 설정
        )

        # print(f"output_ids : {output_ids}")
        # 생성된 텍스트 디코딩
        answer = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        # 질문을 제외한 **답변 부분만 추출**
        #answer = answer.replace(question, "").strip()

        answer = answer.split("대답: ")[-1]
        # answer = answer.replace("\n", "").strip()
        # answer = answer.replace("</s>", "").strip()
        # answer = answer.split("nbsp;")[-1]
        # answer = answer.replace("Question:", "").strip()

        # for to_remove in ["unknown", "#", "ญ", "<b>", "</b>", "</nsp>", "@"]:
        #     answer = answer.replace(to_remove, "")
        # for to_remove in ["2인1조", "2인1조로", "2인 1조", "2인1조의", "2인 1조의", "2인 1조로"]:
        #     answer = answer.replace(to_remove,"")
        # answer = re.sub(r'\(\s*\)', '', answer)

        # # 먼저 answer_list를 만들고
        # answer_list = answer.split("</s>")

        # # answer_list 각 요소에 대해 <, > 제거
        # cleaned_list = []
        # for part in answer_list:
        #     part = part.replace("<", "")
        #     part = part.replace(">", "")
        #     cleaned_list.append(part)

        # 이제 cleaned_list를 통해 result 만들기
        # result = ""
        # for i in list(answer.split(' ')):
        #     if i.endswith('.') or i.endswith(','):
        #         result += i
        
        # answer = ".".join(answer.split(".")[:-1]) + "."

        if answer == "":
            answer = "재발 방지 대책 및 향후 조치 계획"
        
        print(answer)
        test_results.append(answer)



    print("\n테스트 실행 완료! 총 결과 수:", len(test_results))

    # ✅ SentenceTransformer 활용하여 문장 임베딩 생성
    embedding_model_name = "jhgan/ko-sbert-sts"
    embedding = SentenceTransformer(embedding_model_name)

    pred_embeddings = embedding.encode(test_results)
    print(pred_embeddings.shape)  # (샘플 개수, 768)

    # ✅ 최종 결과 CSV 저장
    submission = pd.read_csv(f"{config['data']['dir']}{config['data']['submission_csv']}", encoding="utf-8-sig")

    submission.iloc[:, 1] = test_results  # 모델 생성된 답변 추가
    submission.iloc[:, 2:] = pred_embeddings  # 생성된 문장 임베딩 추가

    submission.to_csv(f"{config['data']['result_dir']}{config['data']['result_file']}", index=False, encoding="utf-8-sig")

    print("✅ 최종 결과 저장 완료!")
