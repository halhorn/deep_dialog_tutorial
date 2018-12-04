# Transformer
この記事では2018年現在 DeepLearning における自然言語処理のデファクトスタンダードとなりつつある Transformer の tf.keras 実装です。
eager mode / graph mode のどちらでも動きます。

## Motivation
[公式の Transformer](https://github.com/tensorflow/models/tree/master/official/transformer) が deprecated な tf.layers ベースで書かれており悲しいので、 tensorflow 2.0 で標準になってくる tf.keras.(layers|models) ベースでの実装を行いました。
私の理解の範囲での、より今後の tensorflow コードとして推奨される形を目指しています。

また、この実装は[作って理解する Transformer / Attention](https://qiita.com/halhorn/private/c91497522be27bde17ce)の教材にもなっています。

## Install
```sh
git clone git@github.com:halhorn/deep_dialog_tutorial.git
cd deep_dialog_tutorial
pip install pipenv
pipenv install
```

## Training
```sh
pipenv run jupyter lab
```
jupyter 上で deepdialog/transformer/training.ipynb を開いてください。
