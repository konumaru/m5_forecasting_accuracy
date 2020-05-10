# Version Reference

実験の最中、Baselineを見失ったのでもう一度はじめから。 


## Plan
1. もう一度 Baselineを引き直す
2. Baselineのコードをリファクタリング
3. 学習時の評価関数として WRMSSE を適応
4. LightGBM の Custom Object を利用して、WRMSSE を最適化するようなものに変更
5. Train データセットに過去のデータになるほど重要でないという重みをつける


Validation Strorategy や 特徴量エンジニアリングは次の実験で行うものとする。


## Experiments
### v03001
