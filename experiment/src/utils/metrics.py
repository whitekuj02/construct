import numpy as np
from transformers import EvalPrediction

def cosine_similarity(a, b):
    """코사인 유사도 계산"""
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product / (norm_a * norm_b) if norm_a != 0 and norm_b != 0 else 0


def jaccard_similarity(text1, text2):
    """자카드 유사도 계산"""
    set1, set2 = set(text1.split()), set(text2.split())  # 단어 집합 생성
    intersection = len(set1.intersection(set2))  # 교집합 크기
    union = len(set1.union(set2))  # 합집합 크기
    return intersection / union if union != 0 else 0

def compute_metrics(eval_pred: EvalPrediction):
    predictions, labels = eval_pred.predictions, eval_pred.label_ids
    
    # 만약 one-hot encoding이라면 argmax로 변환
    if predictions.ndim > 1:
        predictions = np.argmax(predictions, axis=-1)
    
    if labels.ndim > 1:
        labels = np.argmax(labels, axis=-1)

    # 벡터 형태로 처리 가능하도록 변환
    cos_sim = np.mean([cosine_similarity(pred, label) for pred, label in zip(predictions, labels)])
    
    return {"cosine_similarity": cos_sim}