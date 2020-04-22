# Survey Past Competitions

## M5のコンペとしての特徴
- 商品IDがstate_id, store_id, detp_id, cat_idのような階層的な構造を持っている。
- 評価指標は、売上個数 * 売上金額を表現するWRMSSEである。
- べき分布のsalesを予測する。
- 予測データが階層構造に依存しているか。

## 調査対象コンペ

- ASHRAE - Great Energy Predictor III Summary
  - Link：https://www.notion.so/ASHRAE-Great-Energy-Predictor-III-Summary-51d7c29864904ecbb1bd9acc5f424e70
  - 開催時期：4 months ago
  - 評価指標：RMSLE
  - 予測値：各建物におけるメーター種類ごとの消費エネルギー量
- Corporación Favorita Grocery Sales Forecasting
  - Link：https://www.kaggle.com/c/favorita-grocery-sales-forecasting/notebooks
  - 開催時期：2 years ago
  - 評価指標：NWRMSLE, Normalized Weighted Root Mean Squared Logarithmic Error
  - 予測値： Predict Number of visitors
- Recruit Restaurant Visitor Forecasting
  - Link：https://www.kaggle.com/c/recruit-restaurant-visitor-forecasting
  - 開催時期：2 years ago
  - 評価指標：RMSLE
  - 予測値：店舗への来訪者を予測する問題
- Rossmann Store Sales
  - Link：https://www.kaggle.com/c/rossmann-store-sales
  - 開催時期：4 years ago
  - 評価指標：RMSPE（MAPE？）

### Reference
- Related Kaggle time series competitions
    - https://www.kaggle.com/c/m5-forecasting-accuracy/discussion/133474
