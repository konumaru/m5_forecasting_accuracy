# Version Reference

- 特徴量エンジニアリングや目的関数の調査が滞ってきたので、モデリングの方式を変える。


## Experiments
- v05000
  - v04000 の複製
- v05001
  - リファクタリング
- v05002
  - storeごとにモデルを学習し、予測を行う。
- v05003
  - deptごとにモデルを学習し、予測を行う。
- v05004
  - v05002 を踏襲
  - いくつか特徴量を追加する
    - lag 特徴量追加 7, 14,-> 1~14
    - target encoding
- v05005
  - v05003 を踏襲
  - いくつか特徴量を追加する
    - lag 特徴量追加 7, 14,-> 1~14
    - target encoding
- v05006
  - `train_data[train_data['release'] < 30]` を除去
- v05007
  - 学習データを2->3年に変更
  - feature_franction, subsample を 0.5に変更
- v05008（これはだめ）
  - パラメータ調整
    - 'max_bin': 100,
    - 'metric': 'rmse',
    - drop_cols = ['id', 'd', 'sales', 'date', 'wm_yr_wk'] + ['state_id', 'store_id']
- v05009
  - `days_from_last_sales > 7` のデータを捨てる
    - 7日以上欠品が続いているデータを捨てる
- v05010
  - 予測値をlog変換してみる。
- v05011
  - データのnrows削除をなくした。
- v050012
  - v05009 からコピー


### TODO


- リファクタリング
- 特徴量選択
- Hierarchical Bayesian Target Encoding の実装
- 曜日の集約があるなら、wm_yr_wk の集約特徴量もあり
  - これはラグ特徴量も有効
- store_id * dept_id ごとのモデリング
- 同じ特徴料を使っていて、精度が低いのはおかしい。考えられる原因
  - 特徴量の作り方が間違っている
  - 特徴量選択をしていない
  - 実は同じ特徴料ではない
- public LBで使われているデータについても使用する。

