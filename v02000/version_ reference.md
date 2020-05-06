# Version Reference

## v02000でやりたいこと

過去のコンペを調査して、やりたいことがたくさんできた。

その仮設をすべて潰していきたい。

### 過去コンペの調査内容
- 性質の違う予測値を扱う場合、モデルを適切に分ける必要がある。
  - M5でいうと、以下のようなモデルを作る。
    - models every cat_id
    - models every dept_id, cat_id
    - models every store_id, cat_id
    - etc
- 時系列データでもk-Foldは有効


## Versions
### v02000
- baselineの引き直し
- script導入
- Evaluator を外から読み込み


### v02001
- WRMSSE で評価可能
- Evaluator クラスを拡張して、現時点でスコアが改善するsample_weightを生成
  - ひとまず使用せずに、モデルの改善を行う。
- rmse, regression よりも rmse, poisson の方がスコアがよかった。
  - ただし学習時間がかかる

### v02002
- notebookからスクリプトの以降
- validationの手法を少しいじった結果、以下のような知見を得た
  - train_teest_split でもそれっぽく学修したので、SK Foldも有効である可能性が高い
  - ralling, shift の結果nanになった特徴量を捨てないと学修が進まない。
- 上記を踏まえて、v02003で対応する。
- パラメータについて
  - regressionよりもpoissonの方がうまく学習しているようだった。
    - tweedie ってやつもいいらしい、０が非常に多く、残りが正規分布に近づくような分布を仮定する。
