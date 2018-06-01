"""
This module performs unittest on handler module.
"""
import unittest
import sys
sys.path.append('news_analyzer/libraries')
import handler

class TestHandler(unittest.TestCase):
    """
    Usage: handler.py unit-test.
    python test_handler.py
    """

    def setUp(self):
        self.handler = handler.Handler()
        self.query_article = 'SpaceX Launches Rocket'
        self.path = 'news_analyzer/libraries'

    def test_wordcloud(self):
        """
        This method tests get_word_cloud method of handler.py.
        """
        query_wordcloud = self.handler.get_word_cloud(query_article=self.query_article)
        self.assertTrue(query_wordcloud is not None)

    def test_sentiment(self):
        """
        This method tests get_sentiment method of handler.py.
        """
        query_sentiment = self.handler.get_sentiment(query_article=self.query_article)
        self.assertTrue(isinstance(query_sentiment, dict))

    def test_recommended_articles(self):
        """
        This method tests get_recommended_articles method of handler.py.
        """
        recommended_articles = self.handler.get_recommended_articles(
            query_article=self.query_article)
        self.assertTrue(recommended_articles is not None)


if __name__ == '__main__':
    unittest.main()
