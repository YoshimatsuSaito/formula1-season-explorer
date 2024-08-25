# formula1-season-explorer

This app provides dashboards for F1 enthusiasts ([App link](https://formula1-season-explorer.streamlit.app/)). You can view statistics and race winner predictions for the upcoming Grand Prix.

※ This README serves as a personal reference for hobby development purposes. It is primarily intended for use when revisiting or updating the project. It is not designed to instruct or guide a general audience.

## Environment Setup

Install the required packages with the following command: `pip install -r requirements.txt`.

## Application Architecture

This application provides statistics and race winner predictions for the weekend Grand Prix.

- Downloads past results from my AWS S3 bucket.
  - The AWS S3 data source is updated every Monday via a batch process managed by an AWS Lambda function ([repository](https://github.com/YoshimatsuSaito/formula1-basic-info-saver)).
- Creates an in-memory database using DuckDB.
- Generates useful plots to help understand each Grand Prix's features.
- Predicts race winners using a simple LightGBM model.
  - The model is trained on historical data from the 2006 to 2022 seasons, using a [notebook](./notebooks/create_model.ipynb). The trained model is stored in an S3 bucket.
  - The app generates prediction features using the latest data, and the pre-trained model uses these features to predict future race outcomes.

## TODO

- グリッドに関する集計をグランプリ間で比較できるように修正
- 直近数戦の各ドライバーの勢いに関する集計（直近何戦の獲得ポイント比較、ポイント獲得量の傾きなど）
- 予選タイムの接戦度合いの曲線について、グランプリ間で比較できるように修正(ポールと2位のタイム差の中央値比較とかでも良い）
- 各グリッドから、何位くらいまでのレンジでフィニッシュするのかについてのグラフ（レンジや分布を何かしら工夫して見せられれば）
- チームメイトとの差（予選タイム）を秒数で示す図の追加