# src/evaluate.py
import pandas as pd
import pickle
import json
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# 데이터 및 모델 로드
test = pd.read_csv("data/test.csv")
X_test = test.drop("target", axis=1)
y_test = test["target"]

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# 예측
predictions = model.predict(X_test)

# 1. 메트릭 저장 (metrics.json)
acc = accuracy_score(y_test, predictions)
with open("metrics.json", "w") as f:
    json.dump({"accuracy": acc}, f)

# 2. 플롯 저장 (confusion_matrix.png)
cm = confusion_matrix(y_test, predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.savefig("confusion_matrix.png")