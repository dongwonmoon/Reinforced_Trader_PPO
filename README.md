# Reinforced Trader using PPO

---

## 개요

Reinforced Trader using PPO는 기술적 분석과 강화학습의 결합으로 트레이딩 시뮬레이션을 진행하는 프로젝트입니다.  
이 프로젝트는 `ta` 패키지를 활용하여 다양한 기술적 분석 도구들을 feature로 추가하였으며, Transformer 기반의 MLP와 PPO(Proximal Policy Optimization) 알고리즘을 통해 강화학습을 적용합니다.

---

## 프로젝트 특징

- **기술적 분석**:  
  `ta` 패키지를 활용해 다양한 기술적 지표와 분석 도구 제공
- **강화학습**:  
  PPO 알고리즘과 Transformer MLP 구조를 이용한 트레이딩 전략 최적화

---

## Train
python train.py

## Todo

Visualization
트레이딩 결과 및 강화학습 진행 상황을 시각적으로 표현

Analysis
결과 데이터 분석 및 개선사항 도출

References
PPO Implementation:
Barhate, Nikhil (2021). Minimal PyTorch Implementation of Proximal Policy Optimization.
GitHub Repository