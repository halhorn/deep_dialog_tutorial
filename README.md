# Deep Dialog Tutorial
会話モデルを試して見るための簡単なチュートリアルを作ろうと思ったけど RNNLM 作って力尽きたやつ。

# Setup
python は python3 を想定してます。

```zsh
git clone git@github.com:halhorn/deep_dialog_tutorial.git
cd deep_dialog_tutorial
pip install pipenv
pipenv install

# 起動
pipenv run jupyter lab
```

# RNNLM
rnnlm.ipynb

RNN の言語モデル。
たくさんの文章集合から、それっぽい文章を生成するモデルです。

- 学習時：上から順に Train のセクションまで実行してください
- 生成時：Train 以外のそれより上と、 Restore, Generate を実行してください。
    - Restore 時のモデルのパスは適宜変えてください。
