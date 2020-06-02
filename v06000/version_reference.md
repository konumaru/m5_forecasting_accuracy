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
- Evaluation data への対応
- IS_TEST フラグによる処理の制御


## TODO
### v06003
- Hierarchical Bayesian Target Encoding の実装
  - https://www.kaggle.com/konumaru/hierarchical-bayesian-target-encoding
- 上記に合わせて、Target encodingの期間にsubimitデータが含まれていることを修正

### Others
- 特徴量選択
- 曜日の集約があるなら、wm_yr_wk の集約特徴量もあり
  - これはラグ特徴量も有効
- アンサンブル
  - https://www.kaggle.com/mmotoki/generalized-weighted-mean
