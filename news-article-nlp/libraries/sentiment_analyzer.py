"""
sentiment_analyzer

This module has 1 function:
    get_sentiment (see more details below)
"""
from nltk import tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA


def get_sentiment(article):
    """
    A function that analyzes the sentiment on the input string.

    Args:
            input_string (string): string representing the query article.

    Returns:
            Dictionary: A dictionary of sentiment results

    """
    sid = SIA()

    pos_sentences = 0
    neg_sentences = 0
    nue_sentences = 0

    sentences = tokenize.sent_tokenize(article)
    sentences_count = len(sentences)

    for sentence in sentences:
        sentiment = sid.polarity_scores(sentence)

        sentiment_compound = sentiment.get('compound')
        if sentiment_compound > 0:
            pos_sentences += 1
        elif sentiment_compound < 0:
            neg_sentences += 1
        else:
            nue_sentences += 1

    if pos_sentences > neg_sentences:
        overall_sentiment = "Positive"
    elif pos_sentences < neg_sentences:
        overall_sentiment = "Negative"
    else:
        overall_sentiment = "Neutral"

    return {"Overall_Sentiment": overall_sentiment,
            "Positive_Sentences": pos_sentences,
            "Negative_Sentences": neg_sentences,
            "Neutral_Sentences": nue_sentences,
            "Total_Sentences": sentences_count}
