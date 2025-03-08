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

from tqdm import tqdm
# pandas의 apply와 함께 진행 상황 표시를 위해 필요
tqdm.pandas()

class sim_prompt:
    def __init__(self, config: Dict[str, Any]):
        self.config = config

        self.dir = config['data']['dir'] + config['data']['train_csv']
        self.data_df = pd.read_csv(self.dir, encoding='utf-8-sig')

        self.dict = {}
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
                # f"{row['인적사고']} 인적사고"
                # + (f", {row['물적사고']} 물적사고가" if row['물적사고'] != '없음' else "가")
                # + " 발생했습니다. "
                # + f"사고객체는 {row['사고객체(대분류)']} (중분류: {row['사고객체(중분류)']}) "
                # + f"이며, {row['작업프로세스']} 중에 발생한 사고의 원인은 "
                # + f"{row['사고원인']}입니다. 이때, 재발 방지 및 조치 계획을 어떻게 세울까요?"

                # f"건설 현장의 [{row['장소(대)']} - {row['장소(소)']}] 위치에서 "
                # f"{row['공사종류(대분류)']} / {row['공사종류(중분류)']} 공사가 진행 중이었습니다. "
                # f"그 중 {row['공종(대분류)']} (중분류: {row['공종(중분류)']}) 작업을 수행하던 중에 "
                # f"{row['인적사고']} 인적사고"
                # + (f"와 {row['물적사고']} 물적사고가" if row['물적사고'] != '없음' else "가")
                # + " 발생했습니다. "
                # + f"이번 사고의 직접적인 사고객체는 {row['사고객체(대분류)']} "
                # + f"(중분류: {row['사고객체(중분류)']})이며, "
                # + f"부상 부위는 {row['부위(대)']} (소분류: {row['부위(소)']})입니다. "
                # + f"당시 작업프로세스는 '{row['작업프로세스']}'"
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

            answer_text = str(row.get('재발방지대책 및 향후조치계획', ''))

            # 5개의 question-answer 쌍을 리스트로 묶어서 반환
            self.dict[question1] = answer_text
        #    self.dict[question2] = answer_text
        #    self.dict[question3] = answer_text
        #    self.dict[question4] = answer_text
        #    self.dict[question5] = answer_text

        self.data_df.progress_apply(make_prompts_for_row, axis=1)
        print(f"전체 데이터 길이: {len(self.data_df)}")

        # key
        self.keys = list(self.dict.keys())
        print(f"keys 갯수 : {len(self.keys)}")

        self.embedding_model_name = config['data']['rag']['embedding_model']  # 임베딩 모델 선택
        self.embedding = HuggingFaceEmbeddings(model_name=self.embedding_model_name)

        self.vector_store = FAISS.from_texts(self.keys, self.embedding, distance_strategy = DistanceStrategy.COSINE)

        self.retriever = self.vector_store.as_retriever(search_type=config['data']['rag']['search_type'], search_kwargs={**config['data']['rag']['search_kwargs']})
        

    def get_value(self, prompt): #prompt -> dict -> value 
        return self.dict[prompt]

    def get_answer(self, query): # test.csv question -> rag -> prompt
        return self.retriever.get_relevant_documents(query)
    

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
    temp=sim_prompt(config)
    answer = temp.get_answer("건설 현장의 [공동주택 - 내부] 위치에서 건축 / 건축물 공사가 진행 중이었습니다. 그 중 건축 (중분류: 창호 및 유리공사) 작업을 수행하던 중에 넘어짐(물체에 걸림) 인적사고가 발생했습니다. 이번 사고의 직접적인 사고객체는 기타 (중분류: 기타)이며, 부상 부위는 기타 (소분류: 바닥)입니다. 당시 작업프로세스는 '고소작업'였으며, 사고 원인은 사전에 위험요인 제거 미흡.결빙구간 및 장애물을 제거하지 않고 작업.로 파악되었습니다. 이러한 유형의 사고가 재발하지 않도록 방지하기 위한 조치 방안은 무엇일까요?")
    value = temp.get_value(answer[0].page_content)
    # 3개~5개 value 넘겨주자 

    print(answer)
    print(value)