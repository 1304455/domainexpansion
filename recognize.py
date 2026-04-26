"""
領域展開 リアルタイム認識システム
使い方: python recognize.py
"""

import cv2
import mediapipe as mp
import numpy as np
import joblib
import json
import os
import time
from collections import deque

MODEL_DIR = "model"

# ───────────────────────────────────────────────
# 領域展開データ（キャラクター別）
DOMAIN_DATA = {
    "sukuna": {
        "name": "両面宿儺",
        "domain": "伏魔御厨子",
        "color": (0, 30, 180),       # 赤
        "text_color": (80, 80, 255),
        "effect": "■■■ 解・捌 ■■■",
    },
    "gojo": {
        "name": "五条悟",
        "domain": "無量空処",
        "color": (180, 100, 0),      # 青
        "text_color": (255, 180, 50),
        "effect": "∞ 蒼・赫 ∞",
    },
    "jogo": {
        "name": "漏瑚",
        "domain": "蓋棺鉄囲山",
        "color": (0, 100, 200),      # オレンジ
        "text_color": (50, 200, 255),
        "effect": "▲ 隕■ ▲",
    },
    "negative": {
        "name": None,
        "domain": None,
        "color": None,
        "text_color": None,
        "effect": None,
    },
}

# 認識の安定化パラメータ
SMOOTHING_WINDOW = 10    # 直近N フレームで多数決
CONFIDENCE_THRESHOLD = 0.75  # この確信度以上で認識確定
ACTIVATION_HOLD = 1.5    # 秒: 同じジェスチャーを保持したら「発動」
# ───────────────────────────────────────────────


def extract_features(landmarks_list):
    """collect_data.py と同一の特徴量抽出（必ず同期を保つこと）"""
    if len(landmarks_list) < 2:
        return None

    features = []

    for hand_lm in landmarks_list[:2]:
        pts = np.array([[lm.x, lm.y, lm.z] for lm in hand_lm.landmark])
        wrist = pts[0]
        pts = pts - wrist
        scale = np.linalg.norm(pts[9]) + 1e-6
        pts = pts / scale

        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (0, 9), (9, 10), (10, 11), (11, 12),
            (0, 13), (13, 14), (14, 15), (15, 16),
            (0, 17), (17, 18), (18, 19), (19, 20),
        ]
        angles = []
        for (a, b) in connections:
            v = pts[b] - pts[a]
            norm = np.linalg.norm(v) + 1e-6
            angles.extend([v[0] / norm, v[1] / norm, v[2] / norm])

        features.extend(angles)

    wrist0 = np.array([landmarks_list[0].landmark[0].x,
                       landmarks_list[0].landmark[0].y,
                       landmarks_list[0].landmark[0].z])
    wrist1 = np.array([landmarks_list[1].landmark[0].x,
                       landmarks_list[1].landmark[0].y,
                       landmarks_list[1].landmark[0].z])
    rel = wrist1 - wrist0
    scale_rel = (
        np.linalg.norm(np.array([landmarks_list[0].landmark[9].x - wrist0[0],
                                 landmarks_list[0].landmark[9].y - wrist0[1],
                                 landmarks_list[0].landmark[9].z - wrist0[2]]))
        + 1e-6
    )
    features.extend((rel / scale_rel).tolist())
    return np.array(features)


class DomainExpansionApp:
    def __init__(self):
        # モデル読み込み
        clf_path = os.path.join(MODEL_DIR, "classifier.pkl")
        le_path = os.path.join(MODEL_DIR, "label_encoder.pkl")
        meta_path = os.path.join(MODEL_DIR, "meta.json")

        if not os.path.exists(clf_path):
            raise FileNotFoundError(
                "モデルが見つかりません。先に train.py を実行してください。"
            )

        self.pipeline = joblib.load(clf_path)
        self.le = joblib.load(le_path)

        with open(meta_path) as f:
            self.meta = json.load(f)

        print(f"✅ モデル読み込み完了")
        print(f"   クラス: {self.meta['classes']}")
        print(f"   学習精度: {self.meta['cv_accuracy_mean']:.1%}")

        # MediaPipe
        mp_hands = mp.solutions.hands
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
        )
        self.mp_hands = mp_hands
        self.mp_draw = mp.solutions.drawing_utils

        # 認識状態
        self.pred_history = deque(maxlen=SMOOTHING_WINDOW)
        self.conf_history = deque(maxlen=SMOOTHING_WINDOW)
        self.activation_start = None
        self.current_gesture = None
        self.activated_gesture = None
        self.activation_time = None
        self.show_activation = False
        self.activation_display_start = None

    def predict(self, features):
        """特徴量から予測クラスと確信度を返す"""
        feats_2d = features.reshape(1, -1)
        proba = self.pipeline.predict_proba(feats_2d)[0]
        pred_idx = np.argmax(proba)
        pred_label = self.le.inverse_transform([pred_idx])[0]
        confidence = proba[pred_idx]
        return pred_label, confidence, proba

    def smooth_prediction(self, pred, conf):
        """直近Nフレームの多数決で安定化"""
        self.pred_history.append(pred)
        self.conf_history.append(conf)

        if len(self.pred_history) < SMOOTHING_WINDOW // 2:
            return None, 0.0

        # 多数決
        from collections import Counter
        counter = Counter(self.pred_history)
        best_label, best_count = counter.most_common(1)[0]
        ratio = best_count / len(self.pred_history)

        avg_conf = np.mean([c for p, c in zip(self.pred_history, self.conf_history)
                            if p == best_label])
        return best_label, avg_conf * ratio

    def draw_activation_effect(self, frame, gesture_key):
        """領域展開発動エフェクト"""
        h, w = frame.shape[:2]
        data = DOMAIN_DATA[gesture_key]

        # 暗転オーバーレイ
        elapsed = time.time() - self.activation_display_start
        overlay = frame.copy()

        # フェードイン
        alpha = min(elapsed * 2, 0.85)
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # 中央テキスト
        center_x, center_y = w // 2, h // 2
        font = cv2.FONT_HERSHEY_SIMPLEX

        # 領域名（大）
        domain_text = data["domain"]
        text_size = cv2.getTextSize(domain_text, font, 1.5, 3)[0]
        cv2.putText(frame, domain_text,
                    (center_x - text_size[0] // 2, center_y),
                    font, 1.5, data["text_color"], 3)

        # キャラ名（小）
        char_text = data["name"]
        char_size = cv2.getTextSize(char_text, font, 0.9, 2)[0]
        cv2.putText(frame, char_text,
                    (center_x - char_size[0] // 2, center_y + 50),
                    font, 0.9, (200, 200, 200), 2)

        # エフェクトテキスト
        eff_text = data["effect"]
        eff_size = cv2.getTextSize(eff_text, font, 0.8, 2)[0]
        cv2.putText(frame, eff_text,
                    (center_x - eff_size[0] // 2, center_y - 60),
                    font, 0.8, data["text_color"], 2)

        # 「領域展開」テキスト（上部）
        header = "領域展開"
        header_size = cv2.getTextSize(header, font, 1.2, 3)[0]
        cv2.putText(frame, header,
                    (center_x - header_size[0] // 2, 80),
                    font, 1.2, (255, 255, 255), 3)

        return frame

    def run(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ カメラが開けません")
            return

        print("\n[Q] で終了  [R] でリセット\n掌印を結んでください...\n")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = self.hands.process(rgb)

                # 発動エフェクト表示中
                if self.show_activation:
                    elapsed = time.time() - self.activation_display_start
                    if elapsed < 3.0:
                        frame = self.draw_activation_effect(frame, self.activated_gesture)
                        cv2.imshow("領域展開", frame)
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord("q"):
                            break
                        elif key == ord("r"):
                            self.show_activation = False
                        continue
                    else:
                        self.show_activation = False
                        self.activated_gesture = None

                # ランドマーク描画
                if result.multi_hand_landmarks:
                    for hand_lm in result.multi_hand_landmarks:
                        self.mp_draw.draw_landmarks(
                            frame, hand_lm, self.mp_hands.HAND_CONNECTIONS,
                            self.mp_draw.DrawingSpec(color=(0, 255, 100), thickness=2, circle_radius=3),
                            self.mp_draw.DrawingSpec(color=(200, 100, 0), thickness=2),
                        )

                # 予測
                smoothed_label, smoothed_conf = None, 0.0
                if result.multi_hand_landmarks and len(result.multi_hand_landmarks) == 2:
                    feats = extract_features(result.multi_hand_landmarks)
                    if feats is not None:
                        raw_label, raw_conf, proba = self.predict(feats)
                        smoothed_label, smoothed_conf = self.smooth_prediction(raw_label, raw_conf)

                # 発動判定
                if (smoothed_label and smoothed_label != "negative"
                        and smoothed_conf >= CONFIDENCE_THRESHOLD):
                    if smoothed_label != self.current_gesture:
                        self.current_gesture = smoothed_label
                        self.activation_start = time.time()
                    elif time.time() - self.activation_start >= ACTIVATION_HOLD:
                        # 発動！
                        self.activated_gesture = smoothed_label
                        self.show_activation = True
                        self.activation_display_start = time.time()
                        self.current_gesture = None
                        self.activation_start = None
                        self.pred_history.clear()
                        print(f"🔥 領域展開: {DOMAIN_DATA[smoothed_label]['domain']}")
                else:
                    self.current_gesture = None
                    self.activation_start = None

                # UI描画
                h, w = frame.shape[:2]
                cv2.rectangle(frame, (0, 0), (w, 50), (15, 15, 15), -1)
                cv2.rectangle(frame, (0, h - 80), (w, h), (15, 15, 15), -1)

                # 手の検出状態
                hand_count = len(result.multi_hand_landmarks) if result.multi_hand_landmarks else 0
                hc = (0, 200, 80) if hand_count == 2 else (80, 80, 200)
                cv2.putText(frame, f"手: {hand_count}/2", (10, 32),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, hc, 2)

                # 認識結果
                if smoothed_label and smoothed_label != "negative" and smoothed_conf > 0.5:
                    data = DOMAIN_DATA[smoothed_label]
                    label_text = f"{data['name']} ({smoothed_conf:.0%})"
                    cv2.putText(frame, label_text, (120, 32),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.75, data["text_color"], 2)

                    # チャージバー
                    if self.activation_start:
                        charge = min((time.time() - self.activation_start) / ACTIVATION_HOLD, 1.0)
                        bar_w = int((w - 20) * charge)
                        color = (0, int(200 * charge), int(255 * (1 - charge)))
                        cv2.rectangle(frame, (10, h - 70), (10 + bar_w, h - 55), color, -1)
                        cv2.rectangle(frame, (10, h - 70), (w - 10, h - 55), (80, 80, 80), 1)
                        cv2.putText(frame, "CHARGE", (10, h - 40),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)
                        domain_text = DOMAIN_DATA[smoothed_label]["domain"]
                        cv2.putText(frame, domain_text, (10, h - 15),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, data["text_color"], 2)
                else:
                    cv2.putText(frame, "掌印を結んでください", (120, 32),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (120, 120, 120), 1)

                cv2.imshow("領域展開", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("r"):
                    self.pred_history.clear()
                    self.current_gesture = None
                    self.activation_start = None

        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.hands.close()


def main():
    print("\n=== 領域展開 認識システム ===\n")
    try:
        app = DomainExpansionApp()
        app.run()
    except FileNotFoundError as e:
        print(f"\n❌ {e}")


if __name__ == "__main__":
    main()
