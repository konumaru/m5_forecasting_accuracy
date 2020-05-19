# Version Reference

v03000 で Baseline を決めることができた。
ひとまずの目的関数と評価関数を固めたが、目的関数には不安が残った。

一方で特徴量に未着手なので、v04000で一旦、特徴量エンジニアリングを行う


## Plan

- price や calendar の特徴量を追加
  - https://www.kaggle.com/kyakovlev/m5-simple-fe
- lag 特徴量を追加
  - https://www.kaggle.com/kyakovlev/m5-lags-features
- custom features
  - https://www.kaggle.com/kyakovlev/m5-custom-features


## Experiments
### v04000
- Copied from v03004


### v04001: 手当り次第いに特徴量の追加
- calendar 特徴量の追加
- sell_prices 特徴量の追加
  - １年単位のデータは周期性が捉えにくい点や過学習の恐れから捨てた。
- sales_lag_and_roll を追加
  - simple_fe の内容を内包しており、さらに過去に作った特徴量をいくつか追加
- sell_pprice_simple_feature
  - 販売価格の変動率を追加


### v04002
- days_from_last_sales
  - 最後に売上があった日からの経過日数を特徴量に追加
- fillna(-999)
  - 精度下がった。多分nanが多いので、過学習したように思える。
