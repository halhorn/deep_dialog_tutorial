# Deep Dialog Tutorial
会話モデルネタでいろいろ追加していくリポジトリ

- [Transformer](https://github.com/halhorn/deep_dialog_tutorial/tree/master/deepdialog/transformer)
- [RNNLM](https://github.com/halhorn/deep_dialog_tutorial/tree/master/deepdialog/rnnlm)

# Install
python は python3 を想定してます。

```zsh
git clone git@github.com:halhorn/deep_dialog_tutorial.git
cd deep_dialog_tutorial
pip install pipenv
pipenv install

# 起動
pipenv run jupyter lab
```

# Transformer
[コード](https://github.com/halhorn/deep_dialog_tutorial/tree/master/deepdialog/transformer)
[作って理解する Transformer / Attention](https://qiita.com/halhorn/private/c91497522be27bde17ce)

# RNNLM
rnnlm.ipynb

RNN の言語モデル。
たくさんの文章集合から、それっぽい文章を生成するモデルです。

- 学習時：上から順に Train のセクションまで実行してください
- 生成時：Train 以外のそれより上と、 Restore, Generate を実行してください。
    - Restore 時のモデルのパスは適宜変えてください。

# Testing
```py3
./test/run
# or
./test/run deepdialog/transformer/transformer.py
```