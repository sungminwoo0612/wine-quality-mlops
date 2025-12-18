# Wine Quality 예측 모델의 CI/CD 자동화 및 리포팅 시스템

```plaintext
my-mlops-project/
├── .github/
│   └── workflows/
│       └── cml.yaml      # (Step 5) CI/CD 설정
├── data/                 # (Step 2) 데이터 저장소 (Git 무시, DVC 관리)
├── src/
│   ├── prepare.py        # (Step 2) 데이터 로드 및 분할
│   ├── train.py          # (Step 2) 모델 학습
│   └── evaluate.py       # (Step 2) 평가 및 지표 생성
├── dvc.yaml              # (Step 4) 파이프라인 정의
├── params.yaml           # (Step 3) 하이퍼파라미터 설정
├── requirements.txt      # (Step 1) 의존성 라이브러리
└── .dvc/                 # DVC 설정 폴더
```


```
pyenv local 3.11.4
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```