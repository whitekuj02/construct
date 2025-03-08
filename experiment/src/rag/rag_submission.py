# question 에 대한 유사한 prompt 
# sim_prompt
import re
import os
import itertools
import numpy as np
from typing import Any, Dict
import random
import yaml
import argparse
import glob
import torch

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores.utils import DistanceStrategy
import faiss
import itertools

import numpy as np
import pandas as pd 
from typing import Any, Dict
import random
from .sim_prompt import sim_prompt
from tqdm import tqdm
# pandas의 apply와 함께 진행 상황 표시를 위해 필요
tqdm.pandas()

class rag_submission :
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.dir = config['data']['dir'] + config['data']['test_csv']
        self.data_df = pd.read_csv(self.dir, encoding='utf-8-sig')

        self.list = []
        # 컬럼 분리 (안전한 방식으로 적용)
        self.data_df['장소(대)'] = self.data_df['장소'].str.split(' / ').str[0].fillna("")
        self.data_df['장소(소)'] = self.data_df['장소'].str.split(' / ').str[1].fillna("")
        self.data_df['공사종류(대분류)'] = self.data_df['공사종류'].str.split(' / ').str[0].fillna("")
        self.data_df['공사종류(중분류)'] = self.data_df['공사종류'].str.split(' / ').str[1].fillna("")
        self.data_df['공종(대분류)'] = self.data_df['공종'].str.split(' > ').str[0].fillna("")
        self.data_df['공종(중분류)'] = self.data_df['공종'].str.split(' > ').str[1].fillna("")
        self.data_df['사고객체(대분류)'] = self.data_df['사고객체'].str.split(' > ').str[0].fillna("")
        self.data_df['사고객체(중분류)'] = self.data_df['사고객체'].str.split(' > ').str[1].fillna("")
        self.data_df['부위(대)'] = self.data_df['부위'].str.split(' / ').str[0].fillna("")
        self.data_df['부위(소)'] = self.data_df['부위'].str.split(' / ').str[1].fillna("")

        def make_prompts_for_row(row):
            """
            한 행(row)에 대해 5가지 서로 다른 문장 템플릿을 모두 생성하여
            [ {question, answer}, {question, answer}, ... ] 리스트로 반환합니다.
            """

            # 템플릿 1
            question1 = (
                # f"건설 현장의 [{row['장소(대)']} - {row['장소(소)']}] 위치에서 "
                # f"{row['공사종류(대분류)']} / {row['공사종류(중분류)']} 공사가 진행 중이었습니다. "
                # f"그 중 {row['공종(대분류)']} (중분류: {row['공종(중분류)']}) 작업을 수행하던 중에 "
                # f"{row['인적사고']} 인적사고"
                # + (f"와 {row['물적사고']} 물적사고가" if row['물적사고'] != '없음' else "가")
                # + " 발생했습니다. "
                # + f"이번 사고의 직접적인 사고객체는 {row['사고객체(대분류)']} "
                # + f"(중분류: {row['사고객체(중분류)']})이며, "
                # + f"부상 부위는 {row['부위(대)']} (소분류: {row['부위(소)']})입니다. "
                # + f"당시 작업프로세스는 '{row['작업프로세스']}'입니다."
                # + "이러한 유형의 사고가 재발하지 않도록 방지하기 위한 조치 방안은 무엇일까요?"
                f"작업프로세스 '{row['작업프로세스']}' 중 "
                f"{row['사고원인']}으로 인해 {row['인적사고']} 인적사고가 발생했습니다. "
                "이러한 사고를 예방하고 재발을 방지하기 위해 어떤 조치를 취할 수 있을까요?"
            )

            # 템플릿 2
            question2 = (
                f"다음 상황을 가정해 봅시다. "
                + f"작업프로세스 '{row['작업프로세스']}' 도중에 "
                + f"[인적사고: {row['인적사고']} / 물적사고: {row['물적사고']}]가 보고되었습니다. "
                + f"사고객체는 {row['사고객체(대분류)']} (중분류: {row['사고객체(중분류)']})이고 "
                + f"사고 원인은 {row['사고원인']}입니다. "
                + "이러한 사고에 대해 재발을 막고 피해를 최소화하기 위한 방안은 무엇일까요?"
            )

            # 템플릿 3
            question3 = (
                f"작업 현장에서 {row['인적사고']} 인적사고"
                + (f"와 {row['물적사고']} 물적사고가 동시에" if row['물적사고'] != '없음' else "가")
                + " 발생했습니다. "
                + f"사고객체는 {row['사고객체(대분류)']} (중분류: {row['사고객체(중분류)']})이고, "
                + f"사고 원인은 {row['사고원인']}로 파악되었습니다. "
                + f"{row['작업프로세스']} 상황에서 이러한 사고가 발생하지 않도록 하려면 "
                + "어떤 대책과 조치를 마련해야 할까요?"
            )

            # 템플릿 4
            question4 = (
                f"{row['사고원인']} 때문에 "
                + f"{row['인적사고']} 인적사고와 {row['물적사고']} 물적사고가 발생한 사례가 있습니다. "
                + f"당시 작업프로세스는 '{row['작업프로세스']}'였고, "
                + f"사고객체는 {row['사고객체(대분류)']} (중분류: {row['사고객체(중분류)']})였습니다. "
                + "이와 같은 유형의 사고를 예방하기 위해서는 어떤 조치와 방안이 필요할까요?"
            )

            # 템플릿 5
            question5 = (
                f"{row['인적사고']}이(가) 발생하였고 "
                + (f"{row['물적사고']}도 함께 발생한 " if row['물적사고'] != '없음' else "")
                + f"사건입니다. 사고 객체는 {row['사고객체(대분류)']}({row['사고객체(중분류)']})이며 "
                + f"작업프로세스 '{row['작업프로세스']}' 중 발생한 사고의 원인은 "
                + f"'{row['사고원인']}'입니다. "
                + "이에 대한 재발 방지책과 향후 조치 계획을 상세히 제시해 주세요."
            )

            # 1~5 개 중 prompt random 하게 뽑고, 이걸 list에 넣음 , tqdm 써서 progress 확인하기 
         #   data = random.choice([question1, question2, question3, question4, question5]) 
            data = question1
            self.list.append(data)

        self.data_df.progress_apply(make_prompts_for_row, axis=1)
        temp = sim_prompt(config)
        self.result = []

        for i in tqdm(self.list, total=len(self.list), desc="retriver for test.csv"):
            answer = temp.get_answer(i)
            value = temp.get_value(answer[0].page_content)
            self.result.append(value)


        # ✅ SentenceTransformer 활용하여 문장 임베딩 생성
        embedding_model_name = "jhgan/ko-sbert-sts"
        embedding = SentenceTransformer(embedding_model_name)

        pred_embeddings = embedding.encode(self.result)
        print(pred_embeddings.shape)  # (샘플 개수, 768)

        # ✅ 최종 결과 CSV 저장
        submission = pd.read_csv(f"{config['data']['dir']}{config['data']['submission_csv']}", encoding="utf-8-sig")

        submission.iloc[:, 1] = self.result  # 모델 생성된 답변 추가
        submission.iloc[:, 2:] = pred_embeddings  # 생성된 문장 임베딩 추가

        submission.to_csv(f"{config['data']['result_dir']}{config['data']['result_file']}", index=False, encoding="utf-8-sig")

        print("✅ 최종 결과 저장 완료!")
   
    


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

    submission = rag_submission(config)