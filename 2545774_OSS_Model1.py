# <<유방암 데이터셋을 이용한 랜덤 포레스트 분류 모델>>
# <사용 라이브러리: scikit-learn>

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

# [1] 데이터셋 로드
data = load_breast_cancer()
X = data.data   # 입력 특징 (30개)
y = data.target # 레이블 (0: 악성, 1: 양성)

print("[데이터셋 기본 정보]")
print(f"전체 샘플 수 : {len(X)}")
print(f"특징 개수: {X.shape[1]}")
print(f"클래스: {data.target_names.tolist()}")
print(f"악성(0): {sum(y == 0)}개")
print(f"양성(1): {sum(y == 1)}개")

# [2] 학습/테스트 데이터 분리 (8:2)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\n학습 데이터 : {len(X_train)}개")
print(f"테스트 데이터: {len(X_test)}개")

# [3] 모델 학습-기본 파라미터
# n_estimators: 사용할 결정 트리의 개수
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print("\n[모델 1 - 기본 파라미터 (n_estimators=100)]")
print(f"정확도: {acc * 100:.2f}%")
print(classification_report(y_test, y_pred, target_names=data.target_names))

# [4] 하이퍼파라미터 조정 후 재학습
# 트리 수 늘리기 & 과적합 방지를 위해 ---> max_depth 제한
rf_model2 = RandomForestClassifier(
    n_estimators=200,      # 트리 수 증가
    max_depth=10,          # 트리 깊이 제한 ---> 과적합 방지
    min_samples_split=5,   # 분기 조건 강화
    random_state=42
)
rf_model2.fit(X_train, y_train)

y_pred2 = rf_model2.predict(X_test)
acc2 = accuracy_score(y_test, y_pred2)

print("[모델 2 - 파라미터 조정 (n_estimators=200, max_depth=10)]")
print(f"정확도: {acc2 * 100:.2f}%")
print(classification_report(y_test, y_pred2, target_names=data.target_names))

# [5] 두 모델 정확도 비교
print("[모델 성능 비교하기]")
print(f"모델 1 정확도 : {acc * 100:.2f}%")
print(f"모델 2 정확도 : {acc2 * 100:.2f}%")

# [6] 중요 특징 상위 5개 출력
importances = rf_model2.feature_importances_
top5_idx = np.argsort(importances)[::-1][:5]

print("\n[분류에 중요한 특징 Top 5]")
for rank, idx in enumerate(top5_idx, 1):
    print(f"  {rank}위. {data.feature_names[idx]} ({importances[idx]:.4f})")
