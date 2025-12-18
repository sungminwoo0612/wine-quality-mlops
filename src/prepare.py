# src/pandas.py
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
import os

# 데이터 폴더가 없으면 생성
os.makedirs("data", exist_ok=True)

# 와인 데이터 로드
wine = load_wine()
df = pd.DataFrame(wine.data, columns=wine.feature_names)
df['target'] = wine.target

# Train/Test 분리
train, test = train_test_split(df, test_size=0.2, random_state=42)

# 저장
train.to_csv("data/train.csv", index=False)
test.to_csv("data/test.csv", index=False)