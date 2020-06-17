# Version Reference

- PCA or NMF による次元圧縮の処理は未完了
- 一旦 XGBoost による学習 Pipeline と合わせてアンサンブルする準備を行う



## Experiments
### v07000
- v06014 のコピー

### v07001
- LightGBM のラッパーをリファクタリング

### v07002
- [x] XGBoost を用いた学習
  - XGBoost で学習が可能になったが、やはり学習時間が遅い。
  - 一方で精度が上がっているようなん気もする。
  - 環境を整えて、GCP 環境におけるGPUを用いた XGBoost の学習を行うことを検討したい。
- ローカル環境では、学習時間がかかりすぎるので、一旦 LightGBM のみで学習をすすめることを検討する。

### v07003
- [x] LightGBM の Feature Importance を gain に変える
  - https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.plot_importance.html#lightgbm.plot_importance
- [x] Target Encoding に以下を追加
  - ['store_id', 'item_id', 'week'], ['store_id', 'item_id', 'day']
  - 精度は上がったが怪しいので、Target Encoding の範囲から評価期間のデータを野沿いて再学習
    - ちゃんと精度が上がったので採用。


### v07004
- [x] 重みに使ってる scale を sqrt から log1p に変える。
  - 少し精度が上がった。0.5179 -> 0.5129
- [x] 28日前との sales の diff, とその統計量
  - なんかあまり良くない動きをした気がするので不採用 

## Todo
### Ensemble
- [ ] https://www.kaggle.com/mmotoki/generalized-weighted-mean
  - 前処理ですべての値をlog1p 変換する
    - 学習時にlog1pしてうまくいかなかったので、ensenbleでもやらなくてもいいかも？
  - 上記の手法を使って重みを予測
  - Stacking じゃなくていいの？
    - 自分で考えた不安定な手法よりも類似コンペの上位解法を参考にするほうがいいだろう（完）
    - 平均化はこれつかう
      - https://www.kaggle.com/mmotoki/generalized-weighted-mean


### Next Todo
#### 特徴量を増やす
- [ ] いくつかの特徴量の次元圧縮
  - PCAよりNMFのほうが木モデルに使いやすい次元縮約をしてくれる
  - https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.NMF.html
