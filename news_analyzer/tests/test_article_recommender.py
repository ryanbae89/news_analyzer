"""This module does unittest on the functions in article_recommender

Classes:
    TestArticleRecommender: A class of functions to perform
    unit test for article_recommender

Functions:
    test_knn_dimension: A function that checks KDTree
    returns 5 indexes to join to corpus
    test_knn_prediction: A function that checks KDTree
    returns nearest index based on distance
    test_join_type: A function that checks the join_process return type

"""
# standard imports
import sys
import unittest
import numpy as np
import pandas as pd

# pylint: disable=wrong-import-position
sys.path.append('news_analyzer/libraries') # noqa
import configs
import article_recommender

# Set number of articles to return index for as in knn_prediction
NUM_ARTICLES = 5

class TestArticleRecommender(unittest.TestCase):
    """A Class of functions to perform unittest on article_recommender"""
    def setUp(self):
        """
        Initializer for TestArticleRecommender class.
        This function loads the corpus for testing.
        """
        self.corpus = pd.read_csv(configs.CORPUS_PATH)
        self.knn_dimension = article_recommender.knn_prediction(
            np.zeros(shape=(5, 10)), [np.zeros(10)])
        self.knn_logic = article_recommender.knn_prediction(
            np.repeat(np.array([(range(10))]), 10, axis=0).T, [(np.zeros(10))])
        self.join_index = article_recommender.join_process(
            np.array([0, 1, 2]), self.corpus)

    def test_knn_dimension(self):
        """This function checks whether knn_prediction returns 5 nearest indexes
        """
        # Pass in matrix of all zeros for doc_topic_matrix
        # Check knn_prediction returns 5 nearest relevance topics
        self.assertTrue(self.knn_dimension[1].shape[1] == NUM_ARTICLES)

    def test_knn_prediction(self):
        """This function checks whether knn_prediction returns correct nearest indexes
        """
        # Pass in matrix of 0 through 9 for doc_topic_matrix
        # Check all-zero vector is closest to the first row of doc_topic_matrix
        self.assertTrue((self.knn_logic[1] == [0, 1, 2, 3, 4]).all)

    def test_join_type(self):
        """This function check the type of the join process
        """
        self.assertTrue(isinstance(self.join_index, str))


if __name__ == '__main__':
    unittest.main()
