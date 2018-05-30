"""This module takes topic matrix with query article to return recommended articles

Functions:
    knn_prediction: A function that perform KDTree function on topic matrix and query vector
    join_process: A function that joins the first 5 article index back to article corpus
    get_recommended_articles: A function that combines knn_prediction and join_process

"""

# Standard
import numpy as np

# Scikit-Learn imports
from sklearn.neighbors import KDTree

def knn_prediction(doc_topic_matrix, query_vector):
    """ Generate indexes of articles that are close to query article in topic relevance

        Args:
            doc_topic_matrix = probability matrix generated from LDA model on articles corpus
            query_vector = probability vector generated from LDA model on user query article
        Returns:
            dist = distance of the 5 articles in terms of topic relevance
            ind = index of closest 5 articles based on distance in topic relevance
    """
    tree = KDTree(doc_topic_matrix)
    dist, ind = tree.query(np.array(query_vector), k=5)
    return dist, ind

def join_process(index, article_corpus):
    """ Joins index to article corpus to pull out relevant artcile's titles

        Args:
            index = output of knn_prediction denoting index of the recommended articles
            article_corpus = original csv file containing all news articles
        Returns:
            selected_articles = five article titles based on topic relevance
    """
    selected_articles = article_corpus.iloc[index.tolist()[0]]
    return selected_articles['title']

def get_recommended_articles(doc_topic_matrix, query_vector, article_corpus):
    """ Links knn_prediction and join_porcess together and output final result

        Args:
            doc_topic_matrix = probability matrix generated from LDA model on articles corpus
            query_vector = probability vector generated from LDA model on user query article
            article_corpus = original csv file containing all news articles
        Returns:
            selected_articles = five article titles based on topic relevance
    """
    index = knn_prediction(doc_topic_matrix, query_vector)[1]
    return join_process(index, article_corpus)