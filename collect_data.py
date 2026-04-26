"""
領域展開 掌印データ収集ツール
使い方: python collect_data.py --gesture sukuna
対応ジェスチャー: sukuna / gojo / jogo / negative (非掌印)
"""

import cv2
import mediapipe as mp
import numpy as np
import csv
import argparse
import os
import time
from datetime import datetime

# ───────────────────────────────────────────────
GESTURE_LABELS = {
    "sukuna":   "伏魔御厨子（宿儺）",
    "gojo":     "無量空処（五条）",
    "jogo":     "蓋棺鉄囲山（漏瑚）",
    "negative": "非掌印（負例）",
}
DATA_DIR = "data"
SAMPLES_PER_SESSION = 200  # 1セッションで収集するサンプル数
# ───────────────────────────────────────────────


def extract_features(landmarks_list):
    """
    両手のランドマーク21点×2 から特徴量ベクトルを生成。
    スケール・位置不変になるよう正規化する。

    返り値: np.ndarray shape=(126,) または None（手が検出されない場合）
    - 各手の角度特徴量: 20関節角 × 2手 = 40
    - 指先間距離行列: 5指先 × 5指先 × 2手 = 50
    - 両手の相対位置: 手首間ベクトル x,y,z = 3
    - 各手の手のひらサイズ（正規化用）: 2
    - 手の存在フラグ: 2
    合計: 97次元
    """
    # 手が0本 or 1本の場合は特徴抽出不可
    if len(landmarks_list) < 2:
        return None

    features = []

    for hand_lm in landmarks_list[:2]:  # 最大2手
        pts = np.array([[lm.x, lm.y, lm.z] for lm in hand_lm.landmark])

        # 手首を原点に移動
        wrist = pts[0]
        pts = pts - wrist

        # 中指MCP（9番）との距離でスケール正規化
        scale = np.linalg.norm(pts[9]) + 1e-6
        pts = pts / scale

        # 指の関節角度（20個）
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),    # 親指
            (0, 5), (5, 6), (6, 7), (7, 8),    # 人差し指
            (0, 9), (9, 10), (10, 11), (11, 12),  # 中指
            (0, 13), (13, 14), (14, 15), (15, 16),  # 薬指
            (0, 17), (17, 18), (18, 19), (19, 20),  # 小指
        ]
        angles = []
        for (a, b) in connections:
            v = pts[b] - pts[a]
            norm = np.linalg.norm(v) + 1e-6
            # z成分は3D深度。カメラ距離ノイズが大きいのでxy平面の角度も追加
            angles.extend([v[0] / norm, v[1] / norm, v[2] / norm])

        features.extend(angles)  # 20 × 3 = 60次元/手

    # 両手の相対位置（手首間ベクトル）
    wrist0 = np.array([landmarks_list[0].landmark[0].x,
                       landmarks_list[0].landmark[0].y,
                       landmarks_list[0].landmark[0].z])
    wrist1 = np.array([landmarks_list[1].landmark[0].x,
                       landmarks_list[1].landmark[0].y,
                       landmarks_list[1].landmark[0].z])
    rel = wrist1 - wrist0
    # 手全体サイズで正規化（手首→中指MCP距離）
    scale_rel = (
        np.linalg.norm(np.array([landmarks_list[0].landmark[9].x - wrist0[0],
                                 landmarks_list[0].landmark[9].y - wrist0[1],
                                 landmarks_list[0].landmark[9].z - wrist0[2]]))
        + 1e-6
    )
    features.extend((rel / scale_rel).tolist())  # 3次元

    return np.array(features)  # 60×2 + 3 = 123次元


def get_output_path(gesture: str) -> str:
    os.makedirs(DATA_DIR, exist_ok=True)
    return os.path.join(DATA_DIR, f"{gesture}.csv")


def main(gesture: str):
    assert gesture in GESTURE_LABELS, f"未知のジェスチャー: {gesture}"
    label_name = GESTURE_LABELS[gesture]
    out_path = get_output_path(gesture)

    # 既存データのカウント
    existing = 0
    if os.path.exists(out_path):
        with open(out_path) as f:
            existing = sum(1 for _ in f)

    print(f"\n{'='*50}")
    print(f"  収集対象: {label_name}")
    print(f"  既存サンプル数: {existing}")
    print(f"  今回収集目標: {SAMPLES_PER_SESSION}")
    print(f"  保存先: {out_path}")
    print(f"{'='*50}")
    print("\n[Space] で収集開始/停止  [Q] で終了\n")

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ カメラが開けません")
        return

    collecting = False
    count = 0
    csv_file = open(out_path, "a", newline="")
    writer = csv.writer(csv_file)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            # ランドマーク描画
            if result.multi_hand_landmarks:
                for hand_lm in result.multi_hand_landmarks:
                    mp_draw.draw_landmarks(
                        frame, hand_lm, mp_hands.HAND_CONNECTIONS,
                        mp_draw.DrawingSpec(color=(0, 255, 100), thickness=2, circle_radius=4),
                        mp_draw.DrawingSpec(color=(255, 100, 0), thickness=2),
                    )

            # 特徴量抽出 & 保存
            feats = None
            if result.multi_hand_landmarks and len(result.multi_hand_landmarks) == 2:
                feats = extract_features(result.multi_hand_landmarks)

            if collecting and feats is not None:
                writer.writerow([gesture] + feats.tolist())
                csv_file.flush()
                count += 1

            # UI描画
            h, w = frame.shape[:2]
            # 背景バー
            cv2.rectangle(frame, (0, 0), (w, 60), (20, 20, 20), -1)
            cv2.rectangle(frame, (0, h - 70), (w, h), (20, 20, 20), -1)

            status_color = (0, 220, 100) if collecting else (100, 100, 100)
            status_text = "● 収集中" if collecting else "○ 停止中"
            cv2.putText(frame, status_text, (10, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

            cv2.putText(frame, f"{label_name}", (150, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 255), 2)

            hand_count = len(result.multi_hand_landmarks) if result.multi_hand_landmarks else 0
            hand_color = (0, 220, 100) if hand_count == 2 else (0, 100, 220)
            cv2.putText(frame, f"両手検出: {hand_count}/2", (10, h - 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, hand_color, 2)
            cv2.putText(frame, f"収集数: {existing + count} / 目標 {existing + SAMPLES_PER_SESSION}",
                        (10, h - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 220, 220), 2)

            # 進捗バー
            progress = min(count / SAMPLES_PER_SESSION, 1.0)
            bar_w = int((w - 20) * progress)
            cv2.rectangle(frame, (10, h - 65), (10 + bar_w, h - 55), (0, 200, 100), -1)
            cv2.rectangle(frame, (10, h - 65), (w - 10, h - 55), (100, 100, 100), 1)

            cv2.imshow("領域展開 データ収集", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == ord("Q"):
                break
            elif key == ord(" "):
                collecting = not collecting
                if collecting:
                    print(f"  収集開始... ({count}/{SAMPLES_PER_SESSION})")
                else:
                    print(f"  収集停止 (累計: {count})")

            if count >= SAMPLES_PER_SESSION:
                print(f"\n✅ 目標達成！ {count} サンプル収集完了")
                time.sleep(1)
                break

    finally:
        csv_file.close()
        cap.release()
        cv2.destroyAllWindows()
        hands.close()
        print(f"\n保存完了: {out_path} (今回: {count} サンプル)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="掌印データ収集")
    parser.add_argument("--gesture", required=True,
                        choices=list(GESTURE_LABELS.keys()),
                        help="収集するジェスチャー")
    args = parser.parse_args()
    main(args.gesture)
