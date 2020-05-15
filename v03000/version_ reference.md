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


### v03002: Validation に使っているデータを WRMSSE で評価する. 学習時の評価関数として WRMSSE を適応.
- 予測結果をWRMSSE で評価できるようにした
  - 0.50~~~ だった。LBは、0.63296.
- ついで程度だったので、WRMSSE で学習できるようにした。
  - LB 0.63296 -> 0.61504 に改善した。



### v03003: LightGBM のパラメータを変更
- 最新の情報から paramerter を変更
  - Reference: https://www.kaggle.com/kyakovlev/m5-three-shades-of-dark-darker-magic



### v03004: LightGBM の Custom Object を利用して、WRMSSE を最適化するようなものに変更

- custom object を設定したものの精度の向上ができなかった。
  - 参考 Discussion
    - https://www.kaggle.com/c/m5-forecasting-accuracy/discussion/143070
    - https://www.kaggle.com/c/m5-forecasting-accuracy/discussion/138881
  - なにかやり方は違うような気がする。
    - このやり方がでやったといっているっぽい
      - https://www.kaggle.com/sibmike/fast-clear-wrmsse-18ms



このへんで、LightGBM を 2.3.2 に update

### v03005: おまけ、RMSEで最適化
- あまり変化なく最適化された。
- おそらく目的関数が定義できていないので、重みを持った損失で最適化されていない
  - 現状位では、RMSEを最小化するか、WRMSSE を最小化するかの違いしか実装できていない。
  - WRMSSE を最適化する、Tweeide損失関数が必要

### v03006: sales を log1p に変換して予測

- 精度がよくなかった。
- 重みを付けた目的関数でないと、売上０のデータが大量にあり、損失が引っ張られるんだろう。きっと。


## Conclusion
- Baseline を引くことができた。
- RMSE では WRMSSE と同等の最適化ができないことがわかった。
  - しかし、現状はRMSEで最適化したほうがLBのスコアがいい
- 目的関数には、カスタム目的関数を定義するのがいい。
  - しかし現状は、Tweedie による代用が可能（おいおいやらねばならない）
- sample_weight を目的関数や重み付きRMSEなどはうまく機能せず、現状のアプローチが間違っている可能性が高い
  - 他 Kernel を参考にして、正しいやり方を再調査する必要がありそう。
- Baseline を引くことができたので、データクレンジングや特徴量エンジニアリングをしていきたい。
  - Baselineの内容
    - simple_fe
    - tweedie, wrmsse, leave-one-out
