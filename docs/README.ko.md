<div align="center">

# [ATON for Alloy Discovery](https://bin-cao.github.io/AlloyDiscovery/)

**자기일관 추론과 잔차 보정을 결합한 궤적 기반 합금 물성 예측.**

[English](../README.md) | [中文](README.zh-CN.md) | [日本語](README.ja.md) | [한국어](README.ko.md) | [Español](README.es.md) | [Deutsch](README.de.md)

</div>

Alloy Trajectory Optimization Network/Model (ATON/ATOM)은 조성 및 공정 설명자로부터 합금의 기계적 물성을 예측합니다.

- `Strength_MPa`
- `Plasticity_%`

이 모델은 ATOM 신경망 백본, 자기일관 궤적 추론, 내부 잔차 보정 브랜치를 결합합니다. 사용자에게 제공되는 예측 파일에는 최종 ATOM 예측 열만 포함됩니다.

## 결과 미리보기

<p align="center">
  <img src="figs/tensile.png" alt="강도 예측 비교" width="48%">
  <img src="figs/elongation.png" alt="소성 예측 비교" width="48%">
</p>

비교 notebook은 동일한 5-fold out-of-fold 프로토콜에서 ATOM을 RF, GBDT, SVR, Ridge, Lasso, ElasticNet, KNN과 비교합니다.

## 빠른 시작

```bash
python trainer.py
python inference.py input.xlsx predictions.xlsx --checkpoint checkpoints/model_best.pth
```

생성되는 학습 산출물:

```text
checkpoints/model_best.pth
checkpoints/cv_oof_predictions.xlsx
checkpoints/cv_summary.json
```

최종 추론 열:

```text
ATOM_Predicted_Strength_MPa
ATOM_Predicted_Plasticity_%
```

## 저장소 구조

```text
data.xlsx                 학습 데이터
trainer.py                5-fold 교차 검증 학습 진입점
inference.py              라벨 없는 추론 진입점
build_search_space.py     가상 탐색 공간 생성
src/data.py               열 정의 및 데이터셋 구성
src/model.py              ATOM 신경망 모델
src/ensemble.py           내부 잔차 보정 브랜치
src/train.py              학습 및 자기일관 예측 유틸리티
docs/algorithm.html       상세 영어/중국어 알고리즘 문서
docs/README.md            다국어 README 전환 입구
docs/figs/plot.ipynb      5-fold 비교 플롯
```

## 데이터 형식

학습 데이터에는 다음 열이 필요합니다.

```text
Al, Co, Cr, Fe, Ni, Ti, Mo, Nb
Eutectic, Preparation, Processing, Tensile/compress
Cold CR_%, Annealing_Temp_K, Annealing_time_h, Aging_Temp_K, Aging_Time_h
Rate_S-1, Tes_Temp_K
Strength_MPa, Plasticity_%
```

추론 데이터에는 특징 열만 필요합니다. 라벨 열이 있으면 `inference.py`가 자동으로 제거합니다.

## 학습

프로젝트 루트에서 실행합니다.

```bash
python trainer.py
```

교차 검증 결과는 다섯 개 held-out fold 예측을 연결한 뒤 전체 out-of-fold 예측 벡터에서 한 번 평가하여 계산됩니다.

주요 옵션:

```bash
python trainer.py \
  --data data.xlsx \
  --output-dir checkpoints \
  --epochs 300 \
  --batch-size 32 \
  --lr 1e-3 \
  --patience 50 \
  --sc-max-iters 30 \
  --sc-tol 1e-3
```

## 추론

```bash
python inference.py input.xlsx predictions.xlsx --checkpoint checkpoints/model_best.pth
```

## 탐색 공간

```bash
python build_search_space.py --data data.xlsx --output search_space.csv
python inference.py search_space.csv search_predictions.xlsx --checkpoint checkpoints/model_best.pth
```

## 모델 비교 플롯

먼저 모델을 학습합니다.

```bash
python trainer.py
```

그다음 열고 실행합니다.

```text
docs/figs/plot.ipynb
```

## 자체 데이터 사용

1. 필요한 특징 열이 포함된 Excel 또는 CSV 파일을 준비합니다.
2. `Strength_MPa`와 `Plasticity_%`는 학습 데이터에만 유지합니다.
3. 범주형 열을 0 이상의 정수로 인코딩합니다.
4. 원소 열이나 공정 열이 다르면 `src/data.py`의 열 목록을 수정합니다.
5. 학습을 실행합니다.

```bash
python trainer.py --data your_data.xlsx
```

## 문서

전체 알고리즘 가이드:

- [온라인 문서](https://bin-cao.github.io/AlloyDiscovery/)
- [`docs/algorithm.html`](algorithm.html)
