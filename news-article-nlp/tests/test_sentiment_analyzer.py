"""
Unit Tests for Sentiment Analyzer
"""
import sys
sys.path.append('../libraries')
import unittest
import sentiment_analyzer
import matplotlib.pyplot as plt
#from PIL import Image

# Define a class in which the tests will run
class SentimentAnalyzerTest(unittest.TestCase):
    """
    Unit TestCase Class for Sentiment Analyzer
    """

    def test_smoke(self):
        """
        Basic Smoke Test

        Args:
            self (object): Reference to the class

        Returns:
            null
        """
        my_sentiment = sentiment_analyzer.get_sentiment("I am happy.")
        self.assertTrue(my_sentiment is not None)


if __name__ == '__main__':
    unittest.main()