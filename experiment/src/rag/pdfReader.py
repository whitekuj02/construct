import re
import os
import itertools
import numpy as np
from typing import Any, Dict
import random
import yaml
import argparse
import glob

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores.utils import DistanceStrategy
import faiss
import pdfplumber
import torch

class pdf_reader:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.pdf_file_dir = self.config['data']['dir'] + "/" + self.config['data']['rag']['document']
        self.pdf_file_name = [ i for i in os.listdir(self.pdf_file_dir)]
        self.pdf_text = []
        self.count = 0
        for i in self.pdf_file_name:
            pdf_path = f"{self.pdf_file_dir}/{i}"

            # 1. PDF에서 텍스트 추출
            line = self.extract_text_pdfplumber(pdf_path)

            # 2. 전처리된 텍스트
            preprocess_text = self.preprocess_safety_guide(line)

            line_text = preprocess_text.splitlines()
            for k in line_text:
                if len(k) > 30 or k.endswith("다."):
                    self.count += 1
                    self.pdf_text.append(k)

    def extract_text_pdfplumber(self, pdf_path):
        text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text() + "\n"  # 페이지별 텍스트 추출
        return text.strip()

    def preprocess_safety_guide(self, text):
        # 1️⃣ 모든 단락을 한 줄로 변환
        text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)  # 문장 중간 개행 제거

        # 2️⃣ "○", "(가)", "(1)", "1." 같은 소제목 앞에 줄바꿈 추가 \(\w\)|\(\d+\)|
        text = re.sub(r"\s?(○|\d+\. )", r"\n\1", text)

        # 3️⃣ "KOSHA GUIDE" 관련 문서 헤더 제거
        text = re.sub(r"KOSHA GUIDE\s*C\s*-\s*\d+\s*-\s*\d+\s*-?\s*\d*", "", text)
        text = re.sub(r"KOSHA GUIDE\s*[A-Z]-\d{1,3}-\d{4}", "", text)

        # - 7 - 제거
        text = re.sub(r"-\s*\d+\s*-", "", text)

        # 5️⃣ "목차" 제거 ( ·, ─, = 같은 구분선 포함된 줄 삭제)
        text = re.sub(r"^.*[·─=]{10,}.*$", "", text, flags=re.MULTILINE)  # 구분 기호 포함된 줄 삭제

        # # 6️⃣ "그림", "표", "부록" 제거 + 뒤에 이어지는 부가 설명까지 삭제
        # text = re.sub(r"<(그림|표|부록)\s*\d+>", "", text)

        # 7️⃣ 한글이 없는 줄 삭제
        text = re.sub(r"^(?!.*[가-힣]).*$", "", text, flags=re.MULTILINE)

        # ○ 또는 o로 시작하는 라인 제거
        text = re.sub(r"^○.*|^o.*", "", text, flags=re.MULTILINE)

        # 1. 목 적, 2. 적용범위 3. 용어의 정의 제거
        text = re.sub(r"^\d+\.\s*(목\s*적|적용범위|용어의\s*정의).*", "", text, flags=re.MULTILINE)

        # 번호 없는 줄 제거
        text = re.sub(r"^(?!\d+\.\d*).*", "", text, flags=re.MULTILINE)

        # 제목 제거
        text = re.sub(r"^.*한\s*국\s*산\s*업\s*안\s*전\s*보\s*건\s*공\s*단\s*안\s*전\s*보\s*건.*$", "", text, flags=re.MULTILINE)

        # 정규식으로 해당 문장 제거
        text = re.sub(r"^(?!.*(검토|준수|안전|조치|점검|주의|예방|대비|대응|복구|금지)).*$", "", text, flags=re.MULTILINE)
        
        # 빈 줄 제거
        text = re.sub(r"\n+", "\n", text).strip()

        return text
    
    def get_text(self):
        return self.pdf_text
    
    def get_len(self):
        return len(self.pdf_text)
    

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

    cl = pdf_reader(config)
    print(cl.get_text())
    print(cl.get_len())