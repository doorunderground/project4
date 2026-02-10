# viz_ml.py
# ============================================================
# 마스터 노트북(test1.ipynb) 변수명/구조에 맞춘 "지도학습 결과 통합 시각화" 모듈
# - ColumnTransformer 이름: preproc
# - cat transformer 이름: 'cat'
# - num_cols 변수 사용 (feature importance용)
# - Pipeline 구조: ('preproc', preproc), ('classifier', clf)
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)


def _get_score(model, X):
    """
    ROC/PR curve에 사용할 score를 가져옵니다.
    - Pipeline(전처리+모델)도 그대로 처리됩니다.
    """
    # 1) 확률이 있으면 positive class 확률
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        if proba.ndim == 2 and proba.shape[1] >= 2:
            return proba[:, 1]
        return proba

    # 2) decision score가 있으면 사용(SVM 등)
    if hasattr(model, "decision_function"):
        return model.decision_function(X)

    # 3) 마지막 fallback (곡선 품질은 떨어질 수 있음)
    return model.predict(X)


def _safe_feature_names_from_pipeline(model, num_cols=None):
    """
    마스터 노트북 전처리 구조에 맞춰 feature name을 안전하게 추출합니다.
    - model: Pipeline(['preproc', ColumnTransformer], ['classifier', estimator])
    - num_cols: 마스터 노트북의 num_cols (list)
    """
    # 기본값(실패시 None 반환)
    try:
        if not hasattr(model, "__getitem__"):
            return None

        if "preproc" not in model.named_steps:
            return None

        preproc = model.named_steps["preproc"]

        # num 컬럼 이름은 num_cols를 그대로 사용
        base_num = list(num_cols) if num_cols is not None else []

        # cat 컬럼은 OneHotEncoder에서 추출
        # 노트북에서 transformer 이름이 'cat'임
        cat_names = []
        if hasattr(preproc, "named_transformers_") and "cat" in preproc.named_transformers_:
            cat_trans = preproc.named_transformers_["cat"]
            if hasattr(cat_trans, "get_feature_names_out"):
                cat_names = list(cat_trans.get_feature_names_out())

        # 결합
        feature_names = base_num + cat_names
        return feature_names if len(feature_names) > 0 else None

    except Exception:
        return None


def plot_supervised_report(
    model,
    X_test,
    y_test,
    model_name="Model",
    threshold=0.5,
    num_cols=None,
    top_k=10,
):
    """
    [사용 방법]
    - 노트북에서 학습된 Pipeline 모델을 넣으면 됩니다.
      예) model = Pipeline([('preproc', preproc), ('classifier', clf)])
          model.fit(X_train, y_train)
          plot_supervised_report(model, X_test, y_test, model_name="Random Forest", num_cols=num_cols)

    [출력]
    - classification report
    - confusion matrix
    - ROC curve (AUC)
    - PR curve (AP)
    - score distribution
    - feature importance(트리 계열) 또는 coefficient(선형 계열)
    """
    # -------------------------
    # score / prediction
    # -------------------------
    score = _get_score(model, X_test)

    # predict_proba가 있으면 threshold 적용(마스터가 임계값 조절 가능)
    if hasattr(model, "predict_proba"):
        y_pred = (score >= threshold).astype(int)
    else:
        y_pred = model.predict(X_test)

    # -------------------------
    # classification report (텍스트)
    # -------------------------
    print(f"\n{model_name}:")
    print(classification_report(y_test, y_pred))

    # -------------------------
    # 1) Confusion Matrix
    # -------------------------
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5.5, 4.5))
    ConfusionMatrixDisplay(confusion_matrix=cm).plot(values_format="d")
    plt.title(f"{model_name} - Confusion Matrix")
    plt.tight_layout()
    plt.show()

    # -------------------------
    # 2) ROC Curve
    # -------------------------
    try:
        fpr, tpr, _ = roc_curve(y_test, score)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(5.5, 4.5))
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"{model_name} - ROC Curve")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"[{model_name}] ROC Curve 생성 불가: {e}")

    # -------------------------
    # 3) Precision-Recall Curve
    # -------------------------
    try:
        precision, recall, _ = precision_recall_curve(y_test, score)
        ap = average_precision_score(y_test, score)

        plt.figure(figsize=(5.5, 4.5))
        plt.plot(recall, precision, label=f"AP = {ap:.3f}")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"{model_name} - Precision-Recall Curve")
        plt.legend(loc="lower left")
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"[{model_name}] PR Curve 생성 불가: {e}")

    # -------------------------
    # 4) Score distribution
    # -------------------------
    plt.figure(figsize=(6.5, 4))
    plt.hist(score, bins=40)
    plt.title(f"{model_name} - Score Distribution")
    plt.xlabel("score (probability or decision score)")
    plt.ylabel("count")
    plt.tight_layout()
    plt.show()

    # -------------------------
    # 5) Feature importance / coefficient
    # -------------------------
    # Pipeline이면 마지막 estimator는 named_steps['classifier']
    estimator = model
    if hasattr(model, "named_steps") and "classifier" in model.named_steps:
        estimator = model.named_steps["classifier"]

    feature_names = _safe_feature_names_from_pipeline(model, num_cols=num_cols)

    # (A) 트리 계열: feature_importances_
    if hasattr(estimator, "feature_importances_"):
        try:
            importances = np.asarray(estimator.feature_importances_, dtype=float)

            # feature_names 길이가 안 맞으면 인덱스 기반 이름 생성
            if feature_names is None or len(feature_names) != len(importances):
                feature_names = [f"f{i}" for i in range(len(importances))]

            s = (
                pd.Series(importances, index=feature_names)
                .sort_values(ascending=False)
                .head(top_k)
            )

            plt.figure(figsize=(8.5, max(4.5, 0.25 * len(s) + 2)))
            plt.barh(s.index[::-1], s.values[::-1])
            plt.title(f"{model_name} - Feature Importance (Top {top_k})")
            plt.xlabel("importance")
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"[{model_name}] Feature importance 시각화 실패: {e}")

    # (B) 선형 계열: coef_
    elif hasattr(estimator, "coef_"):
        try:
            coef = np.asarray(estimator.coef_, dtype=float)
            if coef.ndim == 2:
                coef = coef[0]

            if feature_names is None or len(feature_names) != len(coef):
                feature_names = [f"f{i}" for i in range(len(coef))]

            s = (
                pd.Series(coef, index=feature_names)
                .sort_values(key=lambda x: np.abs(x), ascending=False)
                .head(top_k)
            )

            plt.figure(figsize=(8.5, max(4.5, 0.25 * len(s) + 2)))
            plt.barh(s.index[::-1], s.values[::-1])
            plt.title(f"{model_name} - Coefficient (Top {top_k}, by |coef|)")
            plt.xlabel("coefficient")
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"[{model_name}] Coef 시각화 실패: {e}")

    else:
        print(f"[{model_name}] 중요도 시각화 불가(지원 속성 없음).")


def fit_and_plot_all(models, preproc, X_train, y_train, X_test, y_test, num_cols, threshold=0.5, top_k=10):
    """
    마스터 노트북의 구조( models dict + preproc + train/test ) 그대로 쓰는 "올인원" 함수.
    - 변수명 건드리지 않고 바로 호출 가능하게 만들었습니다.

    사용 예:
        from viz_ml import fit_and_plot_all
        fit_and_plot_all(models, preproc, X_train, y_train, X_test, y_test, num_cols)

    주의:
    - 내부에서 Pipeline을 새로 만들고 fit 합니다.
    """
    from sklearn.pipeline import Pipeline  # 노트북과 동일 방식

    for name, clf in models.items():
        model = Pipeline([
            ("preproc", preproc),
            ("classifier", clf)
        ])
        model.fit(X_train, y_train)

        plot_supervised_report(
            model=model,
            X_test=X_test,
            y_test=y_test,
            model_name=name,
            threshold=threshold,
            num_cols=num_cols,
            top_k=top_k
        )
