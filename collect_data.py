"""
領域展開 掌印データ収集ツール
使い方:
  両手系: python collect_data.py --gesture sukuna
  片手系: python collect_data.py --gesture gojo
対応ジェスチャー: sukuna / jogo / negative_two / gojo / negative_one
"""

import cv2
import mediapipe as mp
import numpy as np
import csv
import argparse
import os
import time

# ───────────────────────────────────────────────
# モード定義
TWO_HAND_GESTURES = {
    "sukuna":       "伏魔御厨子（宿儺）",
    "jogo":         "蓋棺鉄囲山（漏瑚）",
    "negative_two": "非掌印・両手（負例）",
}
ONE_HAND_GESTURES = {
    "gojo":         "無量空処（五条）",
    "negative_one": "非掌印・右手（負例）",
}
ALL_GESTURES = {**TWO_HAND_GESTURES, **ONE_HAND_GESTURES}

DATA_DIR = "data"
SAMPLES_PER_SESSION = 200
# ───────────────────────────────────────────────


def extract_two_hand_features(landmarks_list):
    """両手用特徴量（123次元）"""
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
        for (a, b) in connections:
            v = pts[b] - pts[a]
            norm = np.linalg.norm(v) + 1e-6
            features.extend([v[0] / norm, v[1] / norm, v[2] / norm])

    wrist0 = np.array([landmarks_list[0].landmark[0].x,
                       landmarks_list[0].landmark[0].y,
                       landmarks_list[0].landmark[0].z])
    wrist1 = np.array([landmarks_list[1].landmark[0].x,
                       landmarks_list[1].landmark[0].y,
                       landmarks_list[1].landmark[0].z])
    rel = wrist1 - wrist0
    scale_rel = (
        np.linalg.norm(np.array([
            landmarks_list[0].landmark[9].x - wrist0[0],
            landmarks_list[0].landmark[9].y - wrist0[1],
            landmarks_list[0].landmark[9].z - wrist0[2],
        ])) + 1e-6
    )
    features.extend((rel / scale_rel).tolist())
    return np.array(features)  # 123次元


def extract_one_hand_features(hand_lm):
    """片手（右手）用特徴量（60次元）"""
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
    features = []
    for (a, b) in connections:
        v = pts[b] - pts[a]
        norm = np.linalg.norm(v) + 1e-6
        features.extend([v[0] / norm, v[1] / norm, v[2] / norm])
    return np.array(features)  # 60次元


def get_right_hand(landmarks_list, handedness_list):
    """handedness情報から右手ランドマークを返す（なければNone）"""
    for hand_lm, handedness in zip(landmarks_list, handedness_list):
        # MediaPipeは映像が反転されているので Left = 実際の右手
        if handedness.classification[0].label == "Left":
            return hand_lm
    return None


def main(gesture: str):
    assert gesture in ALL_GESTURES, f"未知のジェスチャー: {gesture}"
    is_two_hand = gesture in TWO_HAND_GESTURES
    label_name = ALL_GESTURES[gesture]
    mode_str = "両手モード" if is_two_hand else "片手（右手）モード"

    out_path = os.path.join(DATA_DIR, f"{gesture}.csv")
    os.makedirs(DATA_DIR, exist_ok=True)

    existing = 0
    if os.path.exists(out_path):
        with open(out_path) as f:
            existing = sum(1 for _ in f)

    print(f"\n{'='*50}")
    print(f"  収集対象: {label_name}")
    print(f"  モード: {mode_str}")
    print(f"  既存サンプル数: {existing}")
    print(f"  今回収集目標: {SAMPLES_PER_SESSION}")
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

            if result.multi_hand_landmarks:
                for hand_lm in result.multi_hand_landmarks:
                    mp_draw.draw_landmarks(
                        frame, hand_lm, mp_hands.HAND_CONNECTIONS,
                        mp_draw.DrawingSpec(color=(0, 255, 100), thickness=2, circle_radius=4),
                        mp_draw.DrawingSpec(color=(255, 100, 0), thickness=2),
                    )

            # 特徴量抽出
            feats = None
            hand_ok = False

            if is_two_hand:
                if result.multi_hand_landmarks and len(result.multi_hand_landmarks) == 2:
                    feats = extract_two_hand_features(result.multi_hand_landmarks)
                    hand_ok = feats is not None
            else:
                # 片手モード：右手のみ
                if result.multi_hand_landmarks and result.multi_handedness:
                    right_lm = get_right_hand(
                        result.multi_hand_landmarks,
                        result.multi_handedness,
                    )
                    if right_lm is not None:
                        feats = extract_one_hand_features(right_lm)
                        hand_ok = True

            if collecting and hand_ok and feats is not None:
                writer.writerow([gesture] + feats.tolist())
                csv_file.flush()
                count += 1

            # UI
            h, w = frame.shape[:2]
            cv2.rectangle(frame, (0, 0), (w, 60), (20, 20, 20), -1)
            cv2.rectangle(frame, (0, h - 75), (w, h), (20, 20, 20), -1)

            status_color = (0, 220, 100) if collecting else (100, 100, 100)
            cv2.putText(frame, "● 収集中" if collecting else "○ 停止中",
                        (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
            cv2.putText(frame, label_name, (150, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 255), 2)

            # 手の検出状態
            if is_two_hand:
                hand_count = len(result.multi_hand_landmarks) if result.multi_hand_landmarks else 0
                ok = hand_count == 2
                hc = (0, 220, 100) if ok else (0, 100, 220)
                cv2.putText(frame, f"両手検出: {hand_count}/2",
                            (10, h - 48), cv2.FONT_HERSHEY_SIMPLEX, 0.7, hc, 2)
            else:
                ok = hand_ok
                hc = (0, 220, 100) if ok else (0, 100, 220)
                cv2.putText(frame, "右手検出: OK" if ok else "右手検出: NG",
                            (10, h - 48), cv2.FONT_HERSHEY_SIMPLEX, 0.7, hc, 2)

            cv2.putText(frame,
                        f"収集数: {existing + count} / 目標 {existing + SAMPLES_PER_SESSION}",
                        (10, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 220, 220), 2)

            progress = min(count / SAMPLES_PER_SESSION, 1.0)
            bar_w = int((w - 20) * progress)
            cv2.rectangle(frame, (10, h - 68), (10 + bar_w, h - 58), (0, 200, 100), -1)
            cv2.rectangle(frame, (10, h - 68), (w - 10, h - 58), (100, 100, 100), 1)

            cv2.imshow("領域展開 データ収集", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord(" "):
                collecting = not collecting
                print(f"  {'収集開始' if collecting else '収集停止'} (累計: {count})")

            if count >= SAMPLES_PER_SESSION:
                print(f"\n✅ 目標達成！ {count} サンプル収集完了")
                time.sleep(1)
                break

    finally:
        csv_file.close()
        cap.release()
        cv2.destroyAllWindows()
        hands.close()
        print(f"\n保存完了: {out_path} (今回: {count})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gesture", required=True, choices=list(ALL_GESTURES.keys()))
    args = parser.parse_args()
    main(args.gesture)
