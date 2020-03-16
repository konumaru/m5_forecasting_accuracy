# Data Description

## データの関係性。
参考kernel: https://www.kaggle.com/li325040229/eda-and-an-encoder-decoder-lstm-with-9-features

- pricesとsalesの関係性
  - 商品価格が変動しても、売上個数に目立った相関は確認できない。
- eventとsalesの関係性
  - eventの前後にやや売上個数の上昇がみられる。
- カテゴリラベルごとの売上個数
  - ものによってまちまちといった感じ、
  - 各商品のスケールがあっていないので、買い合わせの傾向などは見えにくい。
    - すべてブロットするのではなく、標準化したあとの分散などを出すのもいいかもしれない。
