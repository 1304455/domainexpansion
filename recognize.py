"""
領域展開 リアルタイム認識システム v2
- 宿儺・漏瑚: 両手接近検出トリガー（手が重なる直前の判定を使用）
- 五条:       右手1本、1.5秒保持で発動
使い方: python recognize.py
"""

import cv2
import mediapipe as mp
import numpy as np
import joblib
import json
import os
import time
from collections import deque, Counter

MODEL_DIR = "model"

# ───────────────────────────────────────────────
DOMAIN_DATA = {
    "sukuna": {
        "name": "両面宿儺",
        "domain": "伏魔御厨子",
        "text_color": (80, 80, 255),
        "effect": "■■■ 解・捌 ■■■",
    },
    "gojo": {
        "name": "五条悟",
        "domain": "無量空処",
        "text_color": (255, 180, 50),
        "effect": "∞ 蒼・赫 ∞",
    },
    "jogo": {
        "name": "漏瑚",
        "domain": "蓋棺鉄囲山",
        "text_color": (50, 200, 255),
        "effect": "▲ 隕■ ▲",
    },
}

# 認識パラメータ
SMOOTHING_WINDOW     = 8      # 多数決フレーム数
CONFIDENCE_THRESHOLD = 0.75   # 確信度閾値
ACTIVATION_HOLD      = 1.5    # 五条: 保持秒数
APPROACH_THRESHOLD   = 0.18   # 両手接近判定: 手首間距離（正規化座標）
APPROACH_BUFFER_SEC  = 0.6    # 接近前 何秒分のフレームを遡るか
# ───────────────────────────────────────────────


def extract_two_hand_features(landmarks_list):
    """両手用特徴量（123次元）"""
    if len(landmarks_list) < 2:
        return None
    features = []
    for hand_lm in landmarks_list[:2]:
        pts = np.array([[lm.x, lm.y, lm.z] for lm in hand_lm.landmark])
        pts -= pts[0]
        scale = np.linalg.norm(pts[9]) + 1e-6
        pts /= scale
        connections = [
            (0,1),(1,2),(2,3),(3,4),
            (0,5),(5,6),(6,7),(7,8),
            (0,9),(9,10),(10,11),(11,12),
            (0,13),(13,14),(14,15),(15,16),
            (0,17),(17,18),(18,19),(19,20),
        ]
        for a, b in connections:
            v = pts[b] - pts[a]
            n = np.linalg.norm(v) + 1e-6
            features.extend([v[0]/n, v[1]/n, v[2]/n])
    w0 = np.array([landmarks_list[0].landmark[0].x,
                   landmarks_list[0].landmark[0].y,
                   landmarks_list[0].landmark[0].z])
    w1 = np.array([landmarks_list[1].landmark[0].x,
                   landmarks_list[1].landmark[0].y,
                   landmarks_list[1].landmark[0].z])
    rel = w1 - w0
    sc = np.linalg.norm(np.array([
        landmarks_list[0].landmark[9].x - w0[0],
        landmarks_list[0].landmark[9].y - w0[1],
        landmarks_list[0].landmark[9].z - w0[2],
    ])) + 1e-6
    features.extend((rel / sc).tolist())
    return np.array(features)


def extract_one_hand_features(hand_lm):
    """片手（右手）用特徴量（60次元）"""
    pts = np.array([[lm.x, lm.y, lm.z] for lm in hand_lm.landmark])
    pts -= pts[0]
    scale = np.linalg.norm(pts[9]) + 1e-6
    pts /= scale
    connections = [
        (0,1),(1,2),(2,3),(3,4),
        (0,5),(5,6),(6,7),(7,8),
        (0,9),(9,10),(10,11),(11,12),
        (0,13),(13,14),(14,15),(15,16),
        (0,17),(17,18),(18,19),(19,20),
    ]
    features = []
    for a, b in connections:
        v = pts[b] - pts[a]
        n = np.linalg.norm(v) + 1e-6
        features.extend([v[0]/n, v[1]/n, v[2]/n])
    return np.array(features)


def get_right_hand(landmarks_list, handedness_list):
    """右手ランドマークを返す（映像反転考慮: Label=='Left'が実際の右手）"""
    for lm, hd in zip(landmarks_list, handedness_list):
        if hd.classification[0].label == "Left":
            return lm
    return None


def wrist_distance(lm_list):
    """2手の手首間距離（正規化座標）"""
    w0 = np.array([lm_list[0].landmark[0].x, lm_list[0].landmark[0].y])
    w1 = np.array([lm_list[1].landmark[0].x, lm_list[1].landmark[0].y])
    return np.linalg.norm(w1 - w0)


class DomainExpansionApp:
    def __init__(self):
        # 両手モデル
        two_clf  = os.path.join(MODEL_DIR, "two_hand_classifier.pkl")
        two_le   = os.path.join(MODEL_DIR, "two_hand_label_encoder.pkl")
        # 片手モデル
        one_clf  = os.path.join(MODEL_DIR, "one_hand_classifier.pkl")
        one_le   = os.path.join(MODEL_DIR, "one_hand_label_encoder.pkl")
        meta_path = os.path.join(MODEL_DIR, "meta.json")

        for p in [two_clf, two_le, one_clf, one_le]:
            if not os.path.exists(p):
                raise FileNotFoundError(
                    f"モデルが見つかりません: {p}\n先に train.py を実行してください。"
                )

        self.two_pipeline = joblib.load(two_clf)
        self.two_le       = joblib.load(two_le)
        self.one_pipeline = joblib.load(one_clf)
        self.one_le       = joblib.load(one_le)

        with open(meta_path) as f:
            meta = json.load(f)
        print("✅ モデル読み込み完了")
        print(f"   両手: {meta['two_hand']['classes']}  "
              f"精度 {meta['two_hand']['cv_accuracy_mean']:.1%}")
        print(f"   片手: {meta['one_hand']['classes']}  "
              f"精度 {meta['one_hand']['cv_accuracy_mean']:.1%}")

        mp_hands = mp.solutions.hands
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
        )
        self.mp_hands = mp_hands
        self.mp_draw  = mp.solutions.drawing_utils

        # ── 認識状態 ──
        # 五条用スムージング
        self.one_pred_hist = deque(maxlen=SMOOTHING_WINDOW)
        self.one_conf_hist = deque(maxlen=SMOOTHING_WINDOW)
        # 五条: 発動チャージ
        self.gojo_charge_start = None

        # 両手接近バッファ: (timestamp, label, confidence) を時刻ベースで管理
        # dequeのmaxlenは大きめに取り、古いエントリは時刻で除外する
        self.two_pred_buffer = deque(maxlen=120)  # 最大120フレーム分確保

        # 接近トリガーのクールダウン（連続誤発動防止）
        self.last_activation_time = 0.0
        ACTIVATION_COOLDOWN = 3.0  # 秒: 発動後この時間は再発動しない
        self._cooldown = ACTIVATION_COOLDOWN

        # 発動演出
        self.activated_gesture     = None
        self.activation_disp_start = None

    # ── 予測ヘルパー ──
    def predict_two(self, feats):
        p = self.two_pipeline.predict_proba(feats.reshape(1, -1))[0]
        idx = np.argmax(p)
        return self.two_le.inverse_transform([idx])[0], p[idx]

    def predict_one(self, feats):
        p = self.one_pipeline.predict_proba(feats.reshape(1, -1))[0]
        idx = np.argmax(p)
        return self.one_le.inverse_transform([idx])[0], p[idx]

    def smooth_one(self, label, conf):
        self.one_pred_hist.append(label)
        self.one_conf_hist.append(conf)
        if len(self.one_pred_hist) < SMOOTHING_WINDOW // 2:
            return None, 0.0
        counter = Counter(self.one_pred_hist)
        best, cnt = counter.most_common(1)[0]
        ratio = cnt / len(self.one_pred_hist)
        # 多数決の票率が低い場合はNoneを返す（過剰割り引きせず票率でフィルタ）
        if ratio < 0.5:
            return None, 0.0
        avg_conf = np.mean([c for p, c in zip(self.one_pred_hist, self.one_conf_hist)
                            if p == best])
        return best, avg_conf

    # ── 発動エフェクト ──
    def draw_activation(self, frame, key):
        h, w = frame.shape[:2]
        d = DOMAIN_DATA[key]
        elapsed = time.time() - self.activation_disp_start
        alpha = min(elapsed * 2, 0.85)
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        cx, cy = w // 2, h // 2
        font = cv2.FONT_HERSHEY_SIMPLEX

        for text, y_off, scale, thick in [
            ("領域展開",          -120, 1.2, 3),
            (d["effect"],         -60,  0.8, 2),
            (d["domain"],          0,   1.5, 3),
            (d["name"],            55,  0.9, 2),
        ]:
            sz = cv2.getTextSize(text, font, scale, thick)[0]
            color = (255,255,255) if text in ("領域展開",) else d["text_color"]
            if text == d["name"]:
                color = (200, 200, 200)
            cv2.putText(frame, text, (cx - sz[0]//2, cy + y_off),
                        font, scale, color, thick)
        return frame

    # ── メインループ ──
    def run(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ カメラが開けません")
            return
        print("\n[Q] 終了  [R] リセット\n")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.flip(frame, 1)
                rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = self.hands.process(rgb)

                # 発動演出中
                if self.activated_gesture:
                    if time.time() - self.activation_disp_start < 3.0:
                        frame = self.draw_activation(frame, self.activated_gesture)
                        cv2.imshow("領域展開", frame)
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord("q"): break
                        if key == ord("r"): self.activated_gesture = None
                        continue
                    else:
                        self.activated_gesture = None

                # ランドマーク描画
                lm_list = result.multi_hand_landmarks or []
                hd_list = result.multi_handedness or []
                for lm in lm_list:
                    self.mp_draw.draw_landmarks(
                        frame, lm, self.mp_hands.HAND_CONNECTIONS,
                        self.mp_draw.DrawingSpec(color=(0,255,100), thickness=2, circle_radius=3),
                        self.mp_draw.DrawingSpec(color=(200,100,0), thickness=2),
                    )

                h, w = frame.shape[:2]
                now = time.time()
                status_lines = []  # 画面下部に表示するデバッグ情報

                # ════════════════════════════════
                # ① 両手検出時: バッファ蓄積 + 接近トリガー
                # ════════════════════════════════
                if len(lm_list) == 2:
                    feats = extract_two_hand_features(lm_list)
                    if feats is not None:
                        label, conf = self.predict_two(feats)
                        self.two_pred_buffer.append((now, label, conf))

                    # 古いバッファエントリを時刻で除去（FPS依存を排除）
                    while self.two_pred_buffer and now - self.two_pred_buffer[0][0] > APPROACH_BUFFER_SEC:
                        self.two_pred_buffer.popleft()

                    # 手首間距離を計算
                    dist = wrist_distance(lm_list)
                    status_lines.append(f"両手距離: {dist:.3f}")

                    # 接近トリガー: 閾値以下 かつ クールダウン経過済み かつ バッファに有効な予測あり
                    cooldown_ok = (now - self.last_activation_time) > self._cooldown
                    if dist < APPROACH_THRESHOLD and len(self.two_pred_buffer) > 3 and cooldown_ok:
                        # バッファの直近予測を多数決
                        recent = [(l, c) for _, l, c in self.two_pred_buffer
                                  if l not in ("negative_two",) and c >= CONFIDENCE_THRESHOLD]
                        if recent:
                            counter = Counter(l for l, _ in recent)
                            best_label, cnt = counter.most_common(1)[0]
                            avg_conf = np.mean([c for l, c in recent if l == best_label])
                            if cnt >= 2 and best_label in DOMAIN_DATA:
                                # 発動
                                self.activated_gesture     = best_label
                                self.activation_disp_start = now
                                self.last_activation_time  = now
                                self.two_pred_buffer.clear()
                                print(f"🔥 領域展開（接近トリガー）: {DOMAIN_DATA[best_label]['domain']}")

                # ════════════════════════════════
                # ② 右手1本検出: 五条判定（両手検出中は走らせない）
                # ════════════════════════════════
                right_lm = None
                if len(lm_list) != 2 and lm_list and hd_list:  # 両手検出中は五条判定をスキップ
                    right_lm = get_right_hand(lm_list, hd_list)
                if right_lm is not None:
                    feats1 = extract_one_hand_features(right_lm)
                    raw_label, raw_conf = self.predict_one(feats1)
                    smooth_label, smooth_conf = self.smooth_one(raw_label, raw_conf)

                    if (smooth_label == "gojo"
                            and smooth_conf >= CONFIDENCE_THRESHOLD):
                        if self.gojo_charge_start is None:
                            self.gojo_charge_start = now
                        charge = now - self.gojo_charge_start
                        status_lines.append(f"五条チャージ: {charge:.1f}s")

                        if charge >= ACTIVATION_HOLD:
                            self.activated_gesture     = "gojo"
                            self.activation_disp_start = now
                            self.gojo_charge_start     = None
                            self.one_pred_hist.clear()
                            print("🔥 領域展開（保持トリガー）: 無量空処")
                    else:
                        self.gojo_charge_start = None

                # ════════════════════════════════
                # UI
                # ════════════════════════════════
                cv2.rectangle(frame, (0, 0), (w, 50), (15,15,15), -1)
                cv2.rectangle(frame, (0, h-30*(len(status_lines)+1)), (w, h), (15,15,15), -1)

                hand_count = len(lm_list)
                hc = (0,200,80) if hand_count > 0 else (80,80,200)
                cv2.putText(frame, f"手: {hand_count}本", (10, 32),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, hc, 2)

                # 五条チャージバー
                if self.gojo_charge_start:
                    charge = min((now - self.gojo_charge_start) / ACTIVATION_HOLD, 1.0)
                    bw = int((w - 20) * charge)
                    col = (0, int(200*charge), int(255*(1-charge)))
                    cv2.rectangle(frame, (10, h-50), (10+bw, h-38), col, -1)
                    cv2.rectangle(frame, (10, h-50), (w-10, h-38), (80,80,80), 1)
                    cv2.putText(frame, f"無量空処 CHARGE",
                                (10, h-55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,180,50), 2)

                for i, line in enumerate(status_lines):
                    cv2.putText(frame, line, (10, h - 10 - 22*i),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160,160,160), 1)

                cv2.imshow("領域展開", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"): break
                if key == ord("r"):
                    self.two_pred_buffer.clear()
                    self.one_pred_hist.clear()
                    self.gojo_charge_start = None

        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.hands.close()


if __name__ == "__main__":
    print("\n=== 領域展開 認識システム v2 ===\n")
    try:
        DomainExpansionApp().run()
    except FileNotFoundError as e:
        print(f"\n❌ {e}")
