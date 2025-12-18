# src/train.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle
import yaml

# 파라미터 로드
with open("params.yaml") as f:
    params = yaml.safe_load(f)

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

# 모델 저장
with open("model.pkl", "wb") as f:
    pickle.dump(clf, f)