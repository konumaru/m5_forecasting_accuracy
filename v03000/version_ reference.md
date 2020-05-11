# Version Reference

実験の最中、Baselineを見失ったのでもう一度はじめから。 


## Plan
1. もう一度 Baselineを引き直す
2. Baselineのコードをリファクタリング
3. Validation に使っているデータを WRMSSE で評価する。
4. 学習時の評価関数として WRMSSE を適応
5. LightGBM の Custom Object を利用して、WRMSSE を最適化するようなものに変更
6. Train データセットに過去のデータになるほど重要でないという重みをつける


Validation Strorategy や 特徴量エンジニアリングは次の実験で行うものとする。


## Experiments

### v03000: もう一度 Baselineを引き直す
- Baselineとなりそうなコードをただただスクリプトにしただけ。
- 同じ実行結果がローカル環境でも再現できることを確認
- Reference Kernel
  - Baseline: https://www.kaggle.com/rohitsingh9990/m5-lgbm-fe


### v03001: Baselineのコードをリファクタリング
- v03000 のリファクタリング
- 変更と結果
  - Categorical 変数を指定したら、スコアが、0.60869 -> 0.63296 に下がった。
    - RMSE に向かって最適化した結果、WRMSSE が下がったと思われる。
  - Categorical を指定しつつ、特徴量を追加した結果、0.63296 -> 0.62649 になった。
    - RMSE の世界ではスコアが改善した。
- コードは、だいぶ読みやすくなったので、WRMSSE で評価できるようにする。


### v03002: Validation に使っているデータを WRMSSE で評価する。
