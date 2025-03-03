# 전체적인 구조
- prompt 와 RAG를 이용한 LLM fine-tuning --> 리소스 생각 좀 해야 된다.

# 중요 keyword
1. 각 column간의 correlation 파악 후 feature selection --> 사고에는 많은 유형이 있는데 모든 feature 가 사고로 이어지는 것은 아닐 것!
2. LLM 의 output 을 잘 뽑아 내줄 prompt 실험 --> 전문적인 prompt 가 필요할까?
3. RAG 에 들어갈 chunk vector의 표현력 --> pdf 를 잘 읽고 잘 분해해야 하는 부분
4. LLM 을 효율적으로 fine-tuning 하기 위한 method --> KD, quantization, pruning, Lora
5. LLM 선택 --> scaling 과 knowledge base 를 종합하여 fine tuning 이 쉬울 만한 LLM 을 찾자

리소스에 한계가 있기 때문에 정확도가 높은 LLM 을 한정된 자원 안에서 학습하는 것이 중요
--> 네이버 서버가 28일까지 열려있기 때문에 미리 KD 를 통해 low scale 의 LLM parameter 를 가져오는 것이 핵심이 될 수 있음

# submission embedding model

- from sentence_transformers import SentenceTransformer
- model = SentenceTransformer('jhgan/ko-sbert-sts', use_auth_token=False)

# 평가 metric

- S-Bert Cosine Similartiy (CosineSim) : 의미론적 유사성
- Jaccard Similarity (JaccardSim) : 어휘적 유사성
- 0.7 x ConsineSim + 0.3 x JaccardSim

- 내부 평가의 추론 리소스 평가에 기준이 되는 컴퓨팅 리소스는 A100-80GB X 2대 (Total VRAM 160GB) 이며, 해당 리소스 범위 내에서 모델이 동작할 수 있어야합니다. 