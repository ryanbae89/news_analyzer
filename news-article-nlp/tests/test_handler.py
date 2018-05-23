import unittest
import os
os.chdir('news-article-nlp/libraries')
import handler

class TestHandler(unittest.TestCase):

	def setUp(self):
		self.handler = handler.Handler()
		self.query_article = 'SpaceX Launches Rocket'

	def test_wordcloud(self):
		query_wordcloud = self.handler.get_word_cloud(query_article=self.query_article)
		self.assertTrue(query_wordcloud is not None)

	def test_sentiment(self):
		query_sentiment = self.handler.get_sentiment(query_article=self.query_article)
		self.assertTrue(type(query_sentiment) == dict)

	def test_recommended_articles(self):
		recommended_articles = self.handler.get_recommended_articles(query_article=self.query_article)
		self.assertTrue(recommended_articles is not None)

if __name__ == '__main__':
    unittest.main()