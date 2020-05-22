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


### v04002 - 3
- days_from_last_sales
  - 最後に売上があった日からの経過日数を特徴量に追加
- fillna(-999)
  - 精度下がった。多分nanが多いので、過学習したように思える。



### v04005
- 減衰sample_weightの実装
- 直近６ヶ月は重み１で扱いそこから減衰させる。
- 結果
  - いまのコードでは精度が落ちた。（ローカルでは上がった。）
  - やはり目的関数に調整が必要だろう。

### v04006
- 適切な custom_obj 探し
  - まずは、public kernel に公開されているやつを自分の環境で試してみてもいいかも。
  - custom_asymmetric_train は今の自分の環境では機能しなかった。


## Cunclution
### 特徴量
  - 考えられる特徴量はたくさんがるがメモリや学習速度の関係であまり多くを試すことはできなかった。
  - 更に、今回試した中では最も精度に影響がある特徴量の特定などは困難であった。
    - これが効くんだろうなあ、という特徴量はある。

### モデリング
#### 学習モデル
- tweedie が今回良いと思われる。
  - パラメータの関係でpoisson分布に近いものだと考えられるが一旦tweedieで実験をすすめる。

#### 重み
- 減衰weightは明確に機能したと思われる。
- WRMSSE から算出された weight = sales_ratio / sqrt(scale) の重みは機能しているようにも見える。
  - WRMSSE を損失関数にした場合はそれっぽく学習するが、RMSEの場合はうまくいかない。

#### 損失関数
- やはり、RMSEよりもWRMSSEにしたほうがいいだろう。
  - なので、Cross Validation の戦略も限られた。

### 評価関数
- WRMSSE で評価するべき
  - LBとの関係せがわからなくなる。

### Others
- 実験方法を変える。
  - 今の方法は、並列に実験をできるメリットもあるが、コードの管理や可読性が悪い。
  - １度に入る情報をもっと限定したい。
  - v05000から違う方法でやっていく。
- モデリングに関して。
  - 全データを使った学習方法に限界を感じる。
    - 損失の向かう先が最小化しているか怪しい。
    - store, dept, などのカテゴリごとのモデルを作ったほうがよいだろう。
  - public kernel をもう一度みながら、戦略を練る。
- 決めたこと
  - objective: tweedie
  - metric: WRMSSE
  - weight: (sales_ratio / sqrt(scale)) * decayed_weight
  - CV: time series split (custom)
