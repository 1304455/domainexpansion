# 領域展開 掌印認識システム

呪術廻戦の領域展開の掌印をWebカメラから認識するML分類器。  
MediaPipe による両手ランドマーク追跡 + Gradient Boosting 分類器。

---

## ファイル構成

```
domain_expansion/
├── collect_data.py   # Step 1: データ収集
├── train.py          # Step 2: モデル学習
├── recognize.py      # Step 3: リアルタイム認識
├── data/             # 収集したCSVデータ（自動生成）
│   ├── sukuna.csv
│   ├── gojo.csv
│   ├── jogo.csv
│   └── negative.csv
└── model/            # 学習済みモデル（自動生成）
    ├── classifier.pkl
    ├── label_encoder.pkl
    └── meta.json
```

---

## セットアップ

```bash
pip install mediapipe opencv-python scikit-learn numpy joblib
```

---

## 使い方（3ステップ）

### Step 1: データ収集

各キャラクターの掌印を **200サンプル以上** 収集する。  
`negative`（普通の手の形）も必ず収集すること。

```bash
# 宿儺の伏魔御厨子
python collect_data.py --gesture sukuna

# 五条の無量空処
python collect_data.py --gesture gojo

# 漏瑚の蓋棺鉄囲山
python collect_data.py --gesture jogo

# 非掌印（負例）- 重要！
python collect_data.py --gesture negative
```

**操作:**
- `[Space]` 収集開始/停止
- `[Q]` 終了

**コツ:**
- 両手がフレームに映るように立つ
- 掌印のポーズを取ってから収集開始
- 角度・距離を少しずつ変えながら収集するとモデルが頑健になる
- 各クラス 300 サンプル以上が目安

### Step 2: モデル学習

```bash
python train.py
```

クロスバリデーション精度が **85% 以上** なら実用レベル。

### Step 3: リアルタイム認識

```bash
python recognize.py
```

**操作:**
- 掌印を結んで **1.5秒保持** → 領域展開発動
- `[R]` リセット
- `[Q]` 終了

---

## 認識対象の掌印

| キャラ | 領域展開 | 元ネタ |
|--------|----------|--------|
| 両面宿儺 | 伏魔御厨子 | 閻魔天印 |
| 五条悟 | 無量空処 | 帝釈天印 |
| 漏瑚 | 蓋棺鉄囲山 | 大黒天印 |

---

## アーキテクチャ

```
Webカメラ
  ↓
MediaPipe Hands (両手 21点 × 2)
  ↓
特徴量抽出 (123次元)
  ├─ 指関節の方向ベクトル (60次元/手 × 2)
  └─ 両手の相対位置 (3次元)
  ↓
StandardScaler + Gradient Boosting Classifier
  ↓
スムージング (直近10フレーム多数決)
  ↓
確信度 >= 75% & 1.5秒保持 → 発動
```

### なぜ Gradient Boosting か

- 決定木の組み合わせ → 掌印の「閾値的な」特徴（指が曲がっているか否か等）に強い
- SVM より解釈しやすく、NN より少データで機能する
- ハンドジェスチャー認識のベースラインとして実績あり

---

## 精度改善のヒント

| 問題 | 対策 |
|------|------|
| CV精度が低い | サンプル数を増やす（目標: 各500+） |
| 認識が不安定 | `SMOOTHING_WINDOW` を大きくする |
| 誤検知が多い | `negative` データを増やす |
| 宿儺と漏瑚が混同される | 両手の合わせ方の「指の向き」を意識して収集 |

---

## 拡張案

- **全12キャラ対応**: CSVにキャラ追加 → 収集 → 再学習
- **動作認識**: 印を結ぶ「動き」をLSTMで学習
- **Web版**: MediaPipe JS + TensorFlow.js でブラウザ動作
- **音声エフェクト**: pygame で発動音を追加
