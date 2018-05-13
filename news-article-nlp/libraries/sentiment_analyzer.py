from nltk import tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer as sid

#ToDO: Recalculate sentiment algo
def get_sentiment(article):
    pos_value = 0.0
    neg_value = 0.0

    pos_sentences = 0
    neg_sentences = 0
    nue_sentences = 0

    sentences = tokenize.sent_tokenize(article)
    sentences_count = len(sentences)

    for sentence in sentences:
        sentiment = sid.polarity_scores(sentence)
        pos_value = pos_value + sentiment.get('pos')
        neg_value = neg_value + sentiment.get('neg')

        sentiment_diff = (sentiment.get('pos') - sentiment.get('neg'))
        if sentiment_diff > threshold:
            pos_sentences += 1
        elif sentiment_diff < -threshold:
            neg_sentences += 1
        else:
            nue_sentences += 1

    return {"Overall_Sentiment": "Postive", "Positive_Sentences": pos_sentences, "Negative_Sentences": neg_sentences,
            "Neutral_Sentences": nue_sentences}