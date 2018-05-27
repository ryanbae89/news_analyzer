import unittest
import sys
sys.path.append('news-article-nlp/libraries')
# import 'news-article-nlp'
# print(os.path.join(news-article-nlp.__path__[0]))
import handler

class TestHandler(unittest.TestCase):

	def setUp(self):
		self.handler = handler.Handler()
		self.query_article = 'SpaceX Launches Rocket'
		self.path = 'news-article-nlp/libraries'

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