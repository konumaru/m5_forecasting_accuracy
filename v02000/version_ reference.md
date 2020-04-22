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
