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
def clip_outlier_one_percent(x):
  non_zero_x = np.copy(x)
  non_zero_x = non_zero_x[np.argmax(non_zero_x != 0):]
  lowerbound, upperbound = np.percentile(non_zero_x, [1, 99])
  x = np.clip(x, lowerbound, upperbound).round()
  return x
```

これはだめな前処理だった。
データを歪ませるような処理なのだと思われる。

こういった特徴量を入れるならば、平均値との差などがいいかもしれない。


### v06007
- [x] 休日フラグを作る。
  - 休日で集約した特徴量
  - https://www.kaggle.com/c/m5-forecasting-accuracy/discussion/144842
- [x] sell_price の小数点と整数の分離特徴量


### v06008
- [x] 翌週の休日の数
  - ラベルデータについて気をつけないと行けないかも。
  - カレンダーのデータから生成して、map するほうが良さそう。
- [x] is_not_zero
- [x] id ごとの累積売上個数
  - とりあえず全過去のみ
  - 精度が下がったような、、
  - なので、削った
- [x] is_over_mean
  - Target Encodingを利用する
  - あるカテゴリにおいて、自身（id）は平均よりも高いかどうか

### v06009
- [x] 休日（休日でない日）ごとのターゲットエンコーディング
- [x] Hierarchical Bayesian Target Encoding
   - https://www.kaggle.com/konumaru/hierarchical-bayesian-target-encoding

### v06010
特徴量選択を行う
- [x] melted_and_merged の時点でデータを捨てているが、捨てるのを特徴量エンジニアリングを終えたあとにする
- [ ] 単一の特徴量における分散を用いて特徴量を削減（連続値？）
- [ ] ラベルデータとの相関の小ささを元に特徴量を削減
- [ ] 特徴量同士の相関の強さによって特徴量を削減
- [x] shift, roll 特徴量の欠損値を -999 で保管
  - これは精度が上がった

特徴量選択は、計算コストが重くイマイチうまく行っているのかわからないので今回は省略する。
（モデルの中身を確認するなどはしてもいいかもしれないけど、余裕があればやる。）

### v06011
- [x] 過去のZeroRatioを重みに加えたい
- やる、やらない。
  - 少しスコアが下がったので採用

### v06012
https://lightgbm.readthedocs.io/en/latest/Parameters.html#lambda_l1

- [x] l1 正則化の導入
  - lambda_l1=0.1, スコアが下がった。
- [x] l2 正則化の導入
  - lambda_l2=0.1, 大きく精度が下がらなかった。特徴量選択ができなかったので気休めで採用


### v06013
- [x] 特徴量生成後のデータセットをグループごとに出力する


### v06014
- [x] melted_and_merged_train の時点で、sell_price が null のデータに合わせて sales を null にする
- [x] 上記に合わせて、Target Encoding のコードをリファクタリングする。


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
