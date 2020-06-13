# Version Reference

- PCA or NMF による次元圧縮の処理は未完了
- 一旦 XGBoost による学習 Pipeline と合わせてアンサンブルする準備を行う



## Experiments
### v07000
- v06014 のコピー


## Todo
### 学習方法に関して
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


### Next Todo
#### 特徴量を増やす
- [ ] いくつかの特徴量の次元圧縮
  - PCAよりNMFのほうが木モデルに使いやすい次元縮約をしてくれる
  - https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.NMF.html
- [ ] モデルの中の特徴量重要度を元に特徴量選択について考える
