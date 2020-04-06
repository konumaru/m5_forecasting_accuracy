# Version Reference

## v01000

- Baselineとなるもの
- 少し特徴量エンジニアリングをした。
- まともに学習がうまくいくように、LGBMの評価関数に手を加えた。
    - WRMSEのこと


## v01001

- 少し、LGBMのparameterを調整した。


## v01002

- v01001 でbaseline のスコアを超えなかった。
- validation の日数を28日から90日に増やしてみる。
- model, cv のパラメータをprintするように変更
