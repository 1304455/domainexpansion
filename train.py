"""
領域展開 掌印分類器 トレーニングスクリプト
両手モデル（宿儺・漏瑚）と片手モデル（五条）を別々に学習する。
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

# モデルごとに使うジェスチャーを定義
TWO_HAND_GESTURES = ["sukuna", "jogo", "negative_two"]
ONE_HAND_GESTURES = ["gojo", "negative_one"]


def load_data(gestures: list, expected_dim: int = None) -> tuple:
    X, y = [], []
    counts = {}
    for gesture in gestures:
        path = os.path.join(DATA_DIR, f"{gesture}.csv")
        if not os.path.exists(path):
            print(f"  ⚠️  データなし: {path} (スキップ)")
            continue
        with open(path) as f:
            rows = list(csv.reader(f))
        skipped = 0
        for row in rows:
            if len(row) < 2:
                continue
            try:
                feats = [float(v) for v in row[1:]]
                # 次元チェック: 期待次元と異なるデータを無言で混入させない
                if expected_dim is not None and len(feats) != expected_dim:
                    skipped += 1
                    continue
                X.append(feats)
                y.append(row[0])
            except ValueError:
                continue
        counts[gesture] = len(rows) - skipped
        msg = f"    {gesture}: {counts[gesture]} サンプル"
        if skipped > 0:
            msg += f"  ⚠️ {skipped}件を次元不一致でスキップ"
        print(msg)
    return np.array(X), np.array(y), counts


def train_model(X, y, model_name: str) -> dict:
    """学習してモデルを保存。metricsを返す。"""
    if len(X) == 0:
        print(f"  ❌ {model_name}: データなし")
        return {}

    unique = np.unique(y)
    if len(unique) < 2:
        print(f"  ❌ {model_name}: クラスが1種類しかない ({unique})")
        return {}

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

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

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(pipeline, X, y_enc, cv=cv, scoring="accuracy")
    print(f"  CV Accuracy: {scores.mean():.3f} ± {scores.std():.3f}")

    pipeline.fit(X, y_enc)

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(pipeline, os.path.join(MODEL_DIR, f"{model_name}_classifier.pkl"))
    joblib.dump(le, os.path.join(MODEL_DIR, f"{model_name}_label_encoder.pkl"))

    return {
        "classes": le.classes_.tolist(),
        "n_features": int(X.shape[1]),
        "n_samples": int(len(X)),
        "cv_accuracy_mean": float(scores.mean()),
        "cv_accuracy_std": float(scores.std()),
    }


def main():
    print("\n=== 領域展開 掌印分類器 トレーニング ===\n")

    meta = {}

    # ── 両手モデル（特徴量123次元） ──
    print("【両手モデル】宿儺・漏瑚")
    X2, y2, counts2 = load_data(TWO_HAND_GESTURES, expected_dim=123)
    metrics2 = train_model(X2, y2, "two_hand")
    if metrics2:
        meta["two_hand"] = {**metrics2, "sample_counts": counts2}
        if metrics2["cv_accuracy_mean"] < 0.85:
            print("  ⚠️  精度低め。各クラス300+サンプル推奨")

    print()

    # ── 片手モデル（特徴量60次元） ──
    print("【片手モデル】五条")
    X1, y1, counts1 = load_data(ONE_HAND_GESTURES, expected_dim=60)
    metrics1 = train_model(X1, y1, "one_hand")
    if metrics1:
        meta["one_hand"] = {**metrics1, "sample_counts": counts1}
        if metrics1["cv_accuracy_mean"] < 0.85:
            print("  ⚠️  精度低め。各クラス300+サンプル推奨")

    # メタ保存
    if meta:
        with open(os.path.join(MODEL_DIR, "meta.json"), "w") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        print(f"\n✅ モデル保存完了: {MODEL_DIR}/")
    else:
        print("\n❌ 学習できるデータがありませんでした")


if __name__ == "__main__":
    main()
