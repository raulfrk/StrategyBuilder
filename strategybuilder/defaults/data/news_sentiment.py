from datetime import datetime

from sklearn.pipeline import Pipeline

from otpclient.client.enums import LLMProviderEnum, SentimentAnalysisProcessEnum
from strategybuilder.data.dataframe import DfOp
from strategybuilder.data.fetcher import NewsDataFetch
from strategybuilder.data.filter import FilterNews, FilterSentiment
from strategybuilder.data.sentiment import FlattenSentiment, ExtractSentimentScore
from strategybuilder.data.utils.sentiment_scorer import SumSentimentScorer

SENTIMENT_ANALYSIS_PROCESS = SentimentAnalysisProcessEnum.PLAIN
LLM_PROVIDER = LLMProviderEnum.OLLAMA
SYSTEM_PROMPT = "You are a markets expert. Analyze the sentiment of this financial news related to the given " \
                "symbol and respond with one of the following words about the sentiment [positive, negative, " \
                "neutral]. Respond with only one word."
SENTIMENT_SCORE_AGGREGATION = "1min"


def generate_news_with_sentiment_pipeline(symbol: str,
                                          model: str,
                                          start_time: datetime,
                                          end_time: datetime,
                                          required_headline_content: list[str],
                                          avoid_author: list[str],
                                          sentiment_score_aggregation: str = SENTIMENT_SCORE_AGGREGATION,
                                          model_provider: LLMProviderEnum = LLM_PROVIDER,
                                          sentiment_process: SentimentAnalysisProcessEnum = SENTIMENT_ANALYSIS_PROCESS,
                                          system_prompt: str = SYSTEM_PROMPT,
                                          pick_first_sentiment_on_duplicate: bool = False):
    """
    Generate a pipeline that fetches news data with sentiment.

    :param symbol: The symbol of the stock to fetch news for.
    :param model: The model to use for sentiment analysis.
    :param start_time: The start time of the news data.
    :param end_time: The end time of the news data.
    :param required_headline_content: The required content in the headline (for MSFT an example would be ["microsoft"]).
    :param sentiment_score_aggregation: The aggregation to use for sentiment scores (e.g. "1min").
    :param model_provider: The provider of the model.
    :param sentiment_process: The process to use for sentiment analysis.
    :param system_prompt: The system prompt to use for sentiment analysis.
    :return: The pipeline.
    """
    news_sentiment_pipeline = Pipeline(
        [
            ("sentiment_data_fetch", NewsDataFetch(symbol,
                                                   sentiment_process,
                                                   model=model,
                                                   model_provider=model_provider,
                                                   system_prompt=system_prompt,
                                                   start_time=start_time, end_time=end_time)),
            ("filter_news", FilterNews(lambda x: any(y in x["headline"].lower() for y in required_headline_content))),
            ("filter_author", FilterNews(lambda x: x["author"] not in avoid_author)),
            ("filter_sentiment", FilterSentiment(system_prompt=[system_prompt], symbol=[symbol], failed=False, pick_first_on_duplicate=pick_first_sentiment_on_duplicate)),
            ("flatten_sentiment", FlattenSentiment()),
            ("set_index", DfOp(lambda x: x.set_index("created_at"))),
            ("sentiment_score", ExtractSentimentScore(sentiment_score_aggregation, SumSentimentScorer))
        ]
    )

    return news_sentiment_pipeline
