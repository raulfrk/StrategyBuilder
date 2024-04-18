# StrategyBuilder

The strategy builder is a utilities library that contains all the necessary
tools to build RL-based trading strategies. In particular, it provides the
functionalities through the following packages:

- **data** package: contains data fetchers and data preprocessors (as sklearn
    transformers).
  - **fetcher.py**: contains data fetchers to download market data. It relies on
    the Trading Platform and its python client
  - **dataframe.py**: contains transformers to perform basic operations on dataframes
  - **filter.py**: contains transformers to apply filters both on candle data and news data
  - **gap.py**: contains transformers to handle gaps in the candle data
  - **indicator.py**: contains transformers to compute technical indicators (relies on pandas_ta)
  - **parallel.py**: contains transformers perform transformer operations in parallel (to speed things up)
  - **resampler.py**: contains transformers to resample data to different timeframes
  - **scaler.py**: contains transformers to scale data (e.g. MinMaxScaler)
  - **sentiment.py**: contains transformers to process sentiment labels of news headlines
- **rl/env** package: contains the RL trading environments
  - **long_only_env.py**: contains the long-only, day-trading trading environment used to train RL policies
- **rl/agent.py**: contains functions to create the RL-based agents and train them using Maskable PPO algorithm
- **backtesting** package: contains the backtesting utilities
  - **long_only_backtester.py**: contains backtesting functionalities for long-only strategies
- **defaults/data** package: contains default pipelines for data fetching and preprocessing
  - **candle_data_indicators.py**: contains the default pipeline for fetching candle data and computing technical indicators
  - **news_sentiment.py**: contains the default pipeline for fetching news data with sentiment scores

## Example

A detailed example of using all the functionalities of the StrategyBuilder can be found in the `examples` jupyter notebook.