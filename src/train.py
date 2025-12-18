# src/train.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import pickle
import yaml
import wandb

# 1. W&B 프로젝트 초기화
wandb.init(project="wine-quality-mlops", entity="sungminwoo")
# entity는 내 W&B 아이디(sungminwoo)를 넣거나 비워둔다.

# 파라미터 로드
with open("params.yaml") as f:
    params = yaml.safe_load(f)

# W&B 설정에 파라미터 저장 (나중에 그래프 그릴 때 사용)
wandb.config.update(params)

# 데이터 로드
train = pd.read_csv("data/train.csv")
X_train = train.drop("target", axis=1)
y_train = train["target"]

# 모델 학습 (params.yaml의 값 사용)
clf = RandomForestClassifier(
    n_estimators=params["train"]["n_estimators"],
    max_depth=params["train"]["max_depth"],
    random_state=42
)
clf.fit(X_train, y_train)

# (옵션) 학습 데이터에 대한 성능도 보고 싶다면 여기서 계산
train_acc = accuracy_score(y_train, clf.predict(X_train))

# 2. 메트릭 로그 남기기
# 원래 evaluate.py에서 하던 걸 여기서 같이 로깅하거나, 
# evaluate.py에서도 wandb.init()을 호출해서 로깅할 수 있습니다.
# 여기서는 편의상 학습 스크립트에서 주요 정보를 남깁니다.
wandb.log({
    "accuracy": train_acc,
    "n_estimators": params["train"]["n_estimators"],
    "max_depth": params["train"]["max_depth"]
})

# 모델 저장
with open("model.pkl", "wb") as f:
    pickle.dump(clf, f)