"""
Unit Tests for Sentiment Analyzer
"""
import sys
sys.path.append('news-article-nlp/libraries')
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
        my_sentiment = sentiment_analyzer.get_sentiment("This is a sentence.")
        self.assertTrue(my_sentiment is not None)

    def test_positive_only(self):
        """
        Test a positive sentence

        Args:
            self (object): Reference to the class

        Returns:
            null
        """
        my_sentiment = sentiment_analyzer.get_sentiment("I am happy.")
        self.assertTrue((my_sentiment["Overall_Sentiment"] == 'Positive') & (my_sentiment["Positive_Sentences"] == 1) & (my_sentiment["Negative_Sentences"] == 0) & (my_sentiment["Neutral_Sentences"] == 0) & (my_sentiment["Total_Sentences"] == 1))

    def test_negative_only(self):
        """
        Test a negative sentence

        Args:
            self (object): Reference to the class

        Returns:
            null
        """
        my_sentiment = sentiment_analyzer.get_sentiment("I am really sad.")
        self.assertTrue((my_sentiment["Overall_Sentiment"] == 'Negative') & (my_sentiment["Positive_Sentences"] == 0) & (my_sentiment["Negative_Sentences"] == 1) & (my_sentiment["Neutral_Sentences"] == 0) & (my_sentiment["Total_Sentences"] == 1))

if __name__ == '__main__':
    unittest.main()