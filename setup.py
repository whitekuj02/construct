import os

base_dir = "."

dirs = [
    f"{base_dir}/data", # 데이터 루트
    f"{base_dir}/analysis", # 데이터 분석
    f"{base_dir}/generate", # 데이터 생성 및 증강
    f"{base_dir}/environment", # 가상 환경 세팅
    f"{base_dir}/experiment", # 실험실
    f"{base_dir}/document", # 자료실
    f"{base_dir}/experiment/configs", # 학습 configs
    f"{base_dir}/experiment/outputs", # 학습 outputs ex) .pt, .csv
    f"{base_dir}/experiment/src", # 소스코드
    f"{base_dir}/experiment/src/datasets", # dataset, dataloader
    f"{base_dir}/experiment/src/models", # model
    f"{base_dir}/experiment/src/models/utils", # criterion, optimizer, scheduler
    f"{base_dir}/experiment/src/utils", # 나머지 tool ex) runner, logger, metrics
]

files = [
    f"{base_dir}/experiment/configs/base.yaml", # base config
    f"{base_dir}/experiment/configs/train.yaml", # train 관련 config 
    f"{base_dir}/experiment/configs/data.yaml", # data 관련 config
    f"{base_dir}/experiment/src/datasets/dataloader.py", # dataloader
    f"{base_dir}/experiment/src/datasets/dataset.py", # dataset
    f"{base_dir}/experiment/src/models/base_model.py", # base model
    f"{base_dir}/experiment/src/models/utils/criterion.py", # loss function 호출기
    f"{base_dir}/experiment/src/models/utils/model.py", # model 호출기
    f"{base_dir}/experiment/src/models/utils/optimizer.py", # optimizer 호출기
    f"{base_dir}/experiment/src/models/utils/scheduler.py", # scherduler 호출기
    f"{base_dir}/experiment/src/utils/logger.py", # wandb 설정
    f"{base_dir}/experiment/src/utils/metrics.py", # score 측정 method
    f"{base_dir}/experiment/src/utils/runner.py", # runner
    f"{base_dir}/experiment/src/train.py", # train 용 runner
    f"{base_dir}/experiment/src/test.py", # inference 용 runner
    f"{base_dir}/experiment/main.py", # main 파일
]

for dir in dirs:
    os.makedirs(dir, exist_ok=True)

for file in files:
    open(file, 'a').close()

