# Version Reference

モデルのアンサンブルを行う。

## Experiments
### v08000
- v07004 のコピー
- dept_id ごとに学習、seed = 42

### v08001
- dept_id ごとに学習、seed = 422

### v08002
- dept_id ごとに学習、seed = 4222

### v08003
- store_id ごとに学習、seed = 4222

### v08003
- store_id ごとに学習、seed = 422

### v08004
- store_id ごとに学習、seed = 42

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
