# m5_forecasting_accuracy

Estimate the unit sales of Walmart retail goods.


## Competition Overview



### Trees
- data/
  - raw: 元データを格納
  - reduced: メモリ消費を少なくした元データを格納
  - submit: サブミッションデータを格納
- docs: 雑多なメモ
- notebook: コンペ全体で扱うnotebook
- vXXXXX:
  - features: 加工の特徴量を格納
  - notebook: 試行錯誤中のスクリプトを格納
  - script.py: notebookから吐き出したスクリプトなど。


## GCP Commands

start instance

```
$ gcloud compute instances start kaggle-instance
```


stop instance
```
$ gcloud compute instances stop kaggle-instance
```



## Commands

- 次の実験用ディレクトリの作成

```
$ poetry run new-exp
```

- データサイズの軽量化
  - 注）入力データのファイル名はハードコードすること。

```
$ poetry run reduce-data
```
