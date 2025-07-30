# 室内を自走するペンプロッター

このプロジェクトは、Webotsシミュレーション環境において、IMU、エンコーダー、カメラ、コンパスを統合し、HorizonNetによる室内レイアウト推定で自己位置・向きを推定しながら、ペンを使ってpoints.csvに定義された経路に沿って床に絵を描くロボットシステムです。

## 🎯 システム概要

### ロボット
- **差動二輪ロボット**: ステッピングモータで駆動

### センサー類
- **IMU (Inertial Measurement Unit)**: 角速度・加速度センサーによる姿勢推定
- **エンコーダー**: 車輪回転量による移動距離計測
- **カメラ**: 環境画像の取得と室内レイアウト認識
- **コンパス**: 絶対方位の取得

### 自己位置推論システム
- **HorizonNet**: 室内レイアウト推定による自己位置・向き推定
- **ハイブリッド位置推定**: オドメトリ + レイアウト推定の組み合わせ
- **定期位置リセット**: 30秒間隔でのレイアウト推定による位置補正

### 描画システム
- **ペン制御**: 経路追従時の自動ペン上げ下げ
- **経路描画**: points.csvに定義された座標に沿った連続描画
- **複数図形描画**: 複数の図形パスを順次描画

## 🚀 動作フロー

1. **初期化**: ロボットは原点から外れた位置から開始
2. **初期位置推定**: HorizonNetで自己位置・向きを推定
3. **原点への移動**: 推定された位置から(0,0)に向かって移動
4. **描画開始**: points.csvに定義された経路に沿って絵を描画
   - IMUとエンコーダーによるオドメトリ
   - 各図形パスの開始時にペンを下げる
   - 経路上を移動しながら連続描画
   - 図形パス終了時にペンを上げる
5. **定期位置リセット**: 30秒ごとにHorizonNetで位置をリセット

## 📁 ファイル構成

```
autonomous_pen_plotter/
├── controllers/
│   ├── camera_controller/
│   │   └── camera_controller.py          # 画面撮影用カメラコントローラー
│   └── robot_controller/
│       ├── robot_controller.py           # ロボット制御プログラム
│       ├── inference.py                  # HorizonNet推論処理
│       ├── preprocess.py                 # 画像前処理
│       ├── requirements.txt              # Python依存関係
│       ├── points.csv                    # 描画経路座標データ
│       ├── room_map.json                 # 部屋レイアウト情報
│       └── ckpt/
│           └── resnet50_rnn__mp3d.pth    # HorizonNet事前学習モデル
├── worlds/
│    └── world.wbt                         # Webotsシミュレーション世界
├── libraries/                            # Webotsライブラリ（未使用）
├── plugins/                              # Webotsプラグイン（未使用）
├── protos/                               # Webotsプロトタイプ定義（未使用）
└── README.md                             # このファイル

```

## 🛠️ セットアップ

### 必要なファイル

- `ckpt/resnet50_rnn__mp3d.pth`: HorizonNetの事前学習済みモデル（手動ダウンロードが必要）
- `room_map.json`: 部屋のレイアウト情報をJSONファイルで記述
- `points.csv`: 描画する図形の座標データ

### モデルファイルのダウンロード

HorizonNetの事前学習済みモデル（311.51 MB）は大きすぎるため、GitHubには含まれていません。

1. [HorizonNet GitHub](https://github.com/sunset1995/HorizonNet)から`resnet50_rnn__mp3d.pth`をダウンロード
2. `controllers/robot_controller/ckpt/`ディレクトリに配置

## 🎮 使用方法

### Webotsでの実行

1. Webotsをインストール
2. モデルファイルをダウンロード（上記セットアップ参照）
3. Webotsで`world.wbt`を開く
4. ▶︎ボタンでシミュレーションを開始

### Git LFSについて

このリポジトリはGit LFS（Large File Storage）を使用しています。大きなファイルを扱う場合は、Git LFSのインストールが必要です：

```bash
# macOS
brew install git-lfs
git lfs install

# その他のOS
# https://git-lfs.github.com/ からインストール
```

## ⚙️ パラメータ調整

### 制御パラメータ
- `LAYOUT_UPDATE_INTERVAL`: 位置リセット間隔（秒）
- `INITIAL_POSITION`: 初期位置
- 制御ゲイン: `turn()`と`forward()`メソッド内のパラメータ

### センサーパラメータ
- カメラ視野角: `preprocess.py`内の`fov_h`, `fov_v`
- 車輪パラメータ: `WHEEL_DIAMETER`, `WHEEL_BASE`

## 🔗 参考資料

- [HorizonNet GitHub](https://github.com/sunset1995/HorizonNet): 室内レイアウト推定モデル
- [Webots Documentation](https://cyberbotics.com/doc/): ロボットシミュレーション環境
- [PyTorch Documentation](https://pytorch.org/docs/): ディープラーニングフレームワーク