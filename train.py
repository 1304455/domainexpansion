"""
領域展開 掌印分類器 トレーニングスクリプト
使い方: python train.py
"""

import os
import csv
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib
import json

DATA_DIR = "data"
MODEL_DIR = "model"
GESTURE_LABELS = ["sukuna", "gojo", "jogo", "negative"]


def load_dataset():
    X, y = [], []
    counts = {}

    for gesture in GESTURE_LABELS:
        path = os.path.join(DATA_DIR, f"{gesture}.csv")
        if not os.path.exists(path):
            print(f"⚠️  データなし: {path}")
            continue

        with open(path) as f:
            reader = csv.reader(f)
            rows = list(reader)

        for row in rows:
            if len(row) < 2:
                continue
            label = row[0]
            try:
                feats = [float(v) for v in row[1:]]
                X.append(feats)
                y.append(label)
            except ValueError:
                continue

        counts[gesture] = len(rows)
        print(f"  {gesture}: {len(rows)} サンプル")

    return np.array(X), np.array(y), counts


def train():
    print("\n=== 掌印分類器 トレーニング ===\n")
    print("データ読み込み中...")
    X, y, counts = load_dataset()

    if len(X) == 0:
        print("\n❌ データが見つかりません。先に collect_data.py を実行してください。")
        return

    print(f"\n総サンプル数: {len(X)}")
    print(f"特徴量次元: {X.shape[1]}")

    # クラスが2種類未満はエラー
    unique_labels = np.unique(y)
    if len(unique_labels) < 2:
        print(f"\n❌ 学習には最低2クラスのデータが必要です。現在: {unique_labels}")
        return

    # ラベルエンコード
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    # パイプライン: 標準化 + Gradient Boosting
    # GBはハンドジェスチャーの非線形境界に強い
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", GradientBoostingClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42,
        )),
    ])

    # Stratified K-Fold Cross Validation
    print("\nクロスバリデーション中 (5-fold)...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipeline, X, y_enc, cv=cv, scoring="accuracy")
    print(f"  CV Accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    print(f"  各fold: {[f'{s:.3f}' for s in cv_scores]}")

    # 全データで最終学習
    print("\n全データで最終学習中...")
    pipeline.fit(X, y_enc)

    # 保存
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(pipeline, os.path.join(MODEL_DIR, "classifier.pkl"))
    joblib.dump(le, os.path.join(MODEL_DIR, "label_encoder.pkl"))

    # メタデータ保存
    meta = {
        "classes": le.classes_.tolist(),
        "n_features": X.shape[1],
        "n_samples": len(X),
        "cv_accuracy_mean": float(cv_scores.mean()),
        "cv_accuracy_std": float(cv_scores.std()),
        "sample_counts": counts,
    }
    with open(os.path.join(MODEL_DIR, "meta.json"), "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"\n✅ モデル保存完了: {MODEL_DIR}/")
    print(f"   クラス: {le.classes_.tolist()}")
    print(f"   CV精度: {cv_scores.mean():.1%}")

    if cv_scores.mean() < 0.85:
        print("\n⚠️  精度が低めです。以下を試してください:")
        print("   - 各クラス 300+ サンプル収集")
        print("   - 掌印のバリエーション（角度・距離）を増やす")


if __name__ == "__main__":
    train()
