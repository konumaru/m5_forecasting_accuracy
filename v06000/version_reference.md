# Version Reference

- Evaluation データが配布されたので対応する。
- 特徴量選択について行う
- IS_TEST フラグが必要になったので対応する。
- 積み残しの特徴量について追加する

## Experiments
### v06000
- v05012 のコピー

### v06001
- v05012 のリファクタリング
  - store_id * dept_id ごとのモデリング も行いたい
  - Public LB の結果はあまり良くない。

### v06002
- なにやらパラメータがおかしい様子だったので調整
- モデルをdept_idごとに学習するよう変更

### v06003
- LBとだいたい同じになることを確認
- d を int にする。
- Evaluation data への対応
- IS_TEST フラグによる処理の制御
- できていないこと
  - IS_TEST = False での動作確認

### v06004
- Target Encodingの対象から、ラベルデータの期間を除く
  - ややスコアが上がった
  - `df = df.dropna(subset=['sell_price'])` も必要そうなのでいれてみる。

### v06005
- PC 変えた影響か、精度が変わったので一旦保存

### v06006
- id ごとに sales を 99% cliping
```
upperbound, lowerbound = np.percentile(x, [1, 99])
y = np.clip(x, upperbound, lowerbound)
```


### Others
- アンサンブル
  - https://www.kaggle.com/mmotoki/generalized-weighted-mean
  - いくつかのGroup別に学習したモデル
  - あるGroupで学習したモデルのSEED変更
  - 学習データの期間を変更。（直近2年使う、直近3年間使う、など）
  - 曜日ごとにアンサンブルする
  - Groupごとにアンサンブル
- 特徴量案
  - groupごとに、欠品の数・割合。月ごとなどに集約する
  - 売上金額を考慮した特徴量
    - 過去に作った重み計算用の関数が使えるかも。
  - target encoding するときにlog1p変換をしたときの精度の変化


## Todo
### データクリーニング
- id ごとに sales を 99% cliping
```
upperbound, lowerbound = np.percentile(x, [1, 99])
y = np.clip(x, upperbound, lowerbound)
```

### 特徴量を増やす
- [ ] Hierarchical Bayesian Target Encoding
   - https://www.kaggle.com/konumaru/hierarchical-bayesian-target-encoding
- [ ] 休日フラグを作る。
  - 休日で集約した特徴量
  - https://www.kaggle.com/c/m5-forecasting-accuracy/discussion/144842
- [ ] release, days_from_last_sales の集約特徴量'
- [ ] is_not_zero
- [ ] is_over_mean
- [ ] sell_price の小数点
- [ ] sell_price の切り捨て / 切り上げ
- [ ] id ごとの累積売上個数
- [ ] いくつかの特徴量の次元圧縮
  - PCAよりNMFのほうが木モデルに使いやすい次元縮約をしてくれる
  - https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.NMF.html

### 特徴量選択を行う
- Ridge 回帰を使ってR^2係数から変数の説明力を確認する

### モデルの目的関数についての調査
- これ以上なにも思いつかない

### 学習方法に関して
- [ ] Train データにおいて、Cross Validation をしてみる
  - 普通のCVはできないので、直前28, 56日・１年前の3CVを構築するのはやりたい
- [ ] XGBoost でも学習を行う

### Ensemble
- [ ] https://www.kaggle.com/mmotoki/generalized-weighted-mean
  - 前処理ですべての値をlog1p 変換する
    - 学習時にlog1pしてうまくいかなかったので、ensenbleでもやらなくてもいいかも？
  - 上記の手法を使って重みを予測
  - Stacking じゃなくていいの？
    - 自分で考えた不安定な手法よりも類似コンペの上位解法を参考にするほうがいいだろう（完）
    - 平均化はこれつかう
      - https://www.kaggle.com/mmotoki/generalized-weighted-mean
