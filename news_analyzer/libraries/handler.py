"""
Handle Class

This class handles all the integration between the back-end utilities
and the UI. In particular, this class does all the work to make sure
that what is returned is ready to be visualized in the UI.

This class also persists the data models and corpus for continual
access while the UI is running.
"""
import pickle
import sys
import numpy as np
import pandas as pd
# Internal tools
import configs
import word_cloud_generator
import sentiment_analyzer
import article_recommender
sys.path.append('news_analyzer/libraries')


def get_corpus():
    """
    This method loads the articles corpus csv.
    """
    return pd.read_csv(configs.CORPUS_PATH)


def get_preprocessor():
    """
    This method loads the preprocessed pickle file containing the dtm.
    """
    prep = load_pickled(configs.PREPROCESSOR_PATH)
    return prep


def get_guided_topic_modeler():
    """
    This method loads the guided LDA model.
    """
    modeler = load_pickled(configs.GUIDED_MODELER_PATH)
    return modeler


def get_unguided_topic_modeler():
    """
    This method loads the unguided LDA model.
    """
    modeler = load_pickled(configs.UNGUIDED_MODELER_PATH)
    return modeler


def load_pickled(filename):
    """
    This function loads pickled files.
    """
    with open(filename, "rb") as input_file:
        my_object = pickle.load(input_file)
    return my_object


class Handler():
    """
    Usage: Call the handler
    """
    def __init__(self):
        """
        Initializer for handler class. This function loads the models and
        creates all objects needed to do analysis/visualization.
        """
        self.guided_topic_model = get_guided_topic_modeler()
        self.unguided_topic_model = get_unguided_topic_modeler()
        self.preprocessor = get_preprocessor()
        self.corpus = get_corpus()

    def get_topics(self, query_article):
        """
        Processes query article for recommender system to get
        top query topics.

        Args:
            query_article = string, article or words being queried
        Returns:
            query_topics = list, top guided and all unguided topics
                            distribution
        """
        num_topics = 5
        # convert query article to doc-term-matrix
        query_dtm = self.preprocessor.transform(query_article)
        # join the query article doc-term-matrix with models
        query_guided_topics = self.guided_topic_model.transform(
            np.array(query_dtm))
        query_unguided_topics = self.unguided_topic_model.transform(
            np.array(query_dtm))
        # filter and order topics and percentages
        top_guided_topic_index = query_guided_topics.argsort()[0][-num_topics:]
        guided_topics = np.asarray(
            configs.GUIDED_LDA_TOPICS)[top_guided_topic_index][::-1]
        guided_probs = np.asarray(
            query_guided_topics[0][top_guided_topic_index])[::-1]
        guided_probs = np.round(guided_probs*100, 1)
        guided_probs = guided_probs
        guided_lda_return = np.stack((guided_topics, guided_probs)).T
        # return the topic distribution of the query article for recommender
        query_topics = [guided_lda_return,
                        query_unguided_topics]
        return query_topics

    def get_recommended_articles(self, query_article):
        """
        Processes query article to retrieve top recommended articles.

        Args:
            query_article = string, article or words being queried
        Returns:
            recommended_articles = top recommended articles.
        """
        query_vector = self.get_topics(query_article)[1]
        doc_topic_matrix = self.unguided_topic_model.doc_topic_
        recommended_articles = article_recommender.get_recommended_articles(
            doc_topic_matrix, query_vector, self.corpus)
        return recommended_articles

    @staticmethod
    def get_sentiment(query_article):
        """
        Processes query article for sentiment information.

        Args:
            query_article = string, article or words being queried
        Returns:
            sentiment information = dict, positive, neutral and
                                    negative sentence count.
        """
        return sentiment_analyzer.get_sentiment(query_article)

    @staticmethod
    def get_word_cloud(query_article):
        """
        Processes query article to retrieve word cloud image.

        Args:
            query_article = string, article or words being queried
        Returns:
            wordcloud = image, a wordcloud of the provided query_article
        """
        wordcloud = word_cloud_generator.generate_wordcloud(query_article)
        return wordcloud
