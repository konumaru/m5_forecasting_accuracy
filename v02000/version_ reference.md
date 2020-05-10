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


### v02003
- カテゴリカル変数を設定
- 重みを付けて学習


### v02004
- 現状の重みは、subimit時の重みを利用しているので、もっと相応しい重みを作りたい。
  - 具体的には、その日に合った重みを計算するようにする。
- sample_weightを作り直した。
  - これまで
    - Local 評価に使っているweightをそれっぽく使っていた。
  - 新しいやつ
    - d_XX を基準とし、先28日の売上占有率を計算、過去90日のデータからscaleを計算
    - これらを使い `sales_ratio / sqrt(sclae)` をsample_weightとした。
      - sample_weightは、小さすぎる値になっていたので、min_max_scaleを使って実際よりも大きい値にした。相対値は変わらないはず。
    - 次には、**新しい日付ほど重く学習するように、減衰する重みもつけたい。**
