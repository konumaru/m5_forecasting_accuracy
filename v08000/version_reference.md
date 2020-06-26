# Version Reference

モデルのアンサンブルを行う。

## Experiments
### v08000
- v07004 のコピー
- store_id ごとに学習、seed = 42
- sample_weight を 1.3  倍するのをやめた。
  - 0.5145 -> 0.514
- tweedie の parameter を 1.1 にする
  - 少し精度が下がったので1.0にする
- 'boost_from_average': False
  - 少し精度が下がったので不採用


### v08001
- v08000, model by dept_ids, rand_seed=3

### v08002
- v08000, model by store_ids, rand_seed=3

### v08003
- v08000, model by dept_ids, rand_seed=3
- データを１年に削減

### v08004
- v08000, model by dept_ids, rand_seed=3
- データを2年に削減

### v08005
- [ ] v08001, model by dept_ids, rand_seed=10
- [ ] IS_TEST = False

### v08006
- [ ] 特徴量生成前に、out of stock のデータを null にする
  - 7日以上売上個数が0のデータをnullに変換



## Todo
### Ensemble
- [ ] 今の実力ではアンサンブルを行うことができないので、全体平均を使う。


### Next Todo
#### 特徴量を増やす
- [ ] いくつかの特徴量の次元圧縮
  - PCAよりNMFのほうが木モデルに使いやすい次元縮約をしてくれる
  - https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.NMF.html
