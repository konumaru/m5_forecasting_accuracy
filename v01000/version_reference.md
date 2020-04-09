# Version Reference

## v01000
- Baselineとなるもの
- 少し特徴量エンジニアリングをした。
- まともに学習がうまくいくように、LGBMの評価関数に手を加えた。
    - WRMSEのこと


## v01001
- 少し、LGBMのparameterを調整した。
- baselineよりもスコアがやや下がった。


## v01002
- v01001 でbaseline のスコアを超えなかった。
- validation の日数を28日から90日に増やしてみる。
- model, cv のパラメータをprintするように変更


## v01003
- sell_price に特徴量を追加
- calendar を利用し、翌日、翌々日イベント日フラグを作る。
- そのままでは実行できなくなった。
  - メモリ削減のため、コードのリファクタリングを行う。
  - is release フラグを作って、リリース前のデータを除去
