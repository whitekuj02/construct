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
from typing import Any, Dict

from .pdfReader import pdf_reader

class rag:
    def __init__(self, config: Dict[str, Any]):
        self.config = config

        self.embedding_model_name = config['data']['rag']['embedding_model']  # 임베딩 모델 선택
        self.embedding = HuggingFaceEmbeddings(model_name=self.embedding_model_name)

        self.pdf_text = pdf_reader(config).get_text()
        self.vector_store = FAISS.from_texts(self.pdf_text, self.embedding, distance_strategy = DistanceStrategy.COSINE)

        self.retriever = self.vector_store.as_retriever(search_type=config['data']['rag']['search_type'], search_kwargs={**config['data']['rag']['search_kwargs']})

    def get_answer(self, query):
        return self.retriever.get_relevant_documents(query)
    
