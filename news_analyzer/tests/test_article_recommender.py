"""This module does unittest on the functions in article_recommender

Classes:
    TestArticleRecommender: A class of functions to perform
    unit test for article_recommender

Functions:
    test_knn_dimension: A function that checks KDTree
    returns 5 indexes to join to corpus
    test_knn_prediction: A function that checks KDTree
    returns nearest index based on distance

"""
# standard imports
import sys
import unittest
import numpy as np
# test import
sys.path.append('news_analyzer/libraries')
sys.path.append('../libraries')
import article_recommender


class TestArticleRecommender(unittest.TestCase):
    """A Class of functions to perform unittest on article_recommender"""
    def test_knn_dimension(self):
        """This function checks whether knn_prediction returns 5 nearest indexes
        """
        # Pass in matrix of all zeros for doc_topic_matrix
        # Check knn_prediction returns 5 nearest relevance topics
        self.assertTrue(article_recommender.
                        knn_prediction(np.zeros(shape=(5, 10)),
                                       [np.zeros(10)])[1].shape[1] == 5)

    def test_knn_prediction(self):
        """This function checks whether knn_prediction returns correct nearest indexes
        """
        # Pass in matrix of 0 through 9 for doc_topic_matrix
        # Check all-zero vector is closest to the first row of doc_topic_matrix
        self.assertTrue((article_recommender.knn_prediction(
            np.repeat(np.array([(range(10))]), 10, axis=0).T,
            [(np.zeros(10))])[1] == [[0, 1, 2, 3, 4]]).all)


if __name__ == '__main__':
    unittest.main()
