# airliftチーム機械学習レポジトリの使い方

## 執筆者　松原貞徳

データ構造：.psdataが生波形になります。これら信号に対しバンドパスフィルタをかけるといった操作を行うことによって画像を生成したり、あるいは音響特徴量を使って目標変数との物理的なモデル化を行うアプローチがあります。これら生データの場所は研究室データサーバの＜＞にあります。
しかし、これらは機械学習の入力としては扱いづらいことが多いので、使いやすい形式に変更します。具体的には、.npy形式にしてx_data.npy, t_data.npyとして管理します。入力信号及び目標変数の値が紐づけられていて、これらを生成するスクリプトが＜＞にあります。input_path,output_pathを指定することでそれらが生成されます。

```
.
├── configs
│   └── config.yaml
├── data
│   ├── t_train.npy
│   ├── x_test.npy
│   ├── x_train.npy
│   └── y_train.npy
├── eval.py
├── outputs
│   ├── 2025-01-22
│   └── 2025-01-27
├── readme.md
├── requirements.txt
├── setup.py
├── src
│   ├── datasets
│   ├── models
│   ├── preprocess
│   └── utils
├── tmp
│   ├── eval
│   └── train
├── train.py
```
のような構造をしています。
## プロジェクト概要
`setup.py`を実行し、データのロードや訓練がうまくいくのかについて確認してください。`requirements.txt`には必要なモジュールがすべて記載されています。各々の環境にインストールして使用してください。
`configs/config.yaml`にはGPUを何台使用するかといったハードウェアの使い方や、バッチサイズやEpoch数などのパラメータに関する設定が書かれています。
`data`ディレクトリには学習に用いるデータが保存されています。しかしこれらは`.gitignore`ファイルによって追跡が無効化されていることに注意してください。
`output`ディレクトリには、hydraパッケージによって訓練済モデルが作成された日付の名前に応じた名前のフォルダが自動で生成されています。作成されたモデルのパスを指定して`eval.py`を実行することによって、機械学習モデルの性能を評価することができます。
`src`ディレクトリには共通関数が記載されています。これらは`train.py`や`eval.py`の中で`from src.utils.utils import fix_seed`のように関数を呼び出し使用することができるようになっています。`datasets/dataset.py`ではデータローダを定義してモデルへの入力に使用しやすい形となっており、`src/models`の中に訓練すべきモデルが記載されています。これらもプロジェクトに応じて自由に呼び出して使うようにしてください。