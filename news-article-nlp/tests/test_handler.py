import unittest
import sys
sys.path.append('../libraries')
import handler

class TestHandler(unittest.TestCase):

	def test_smoke(self):
		test_handler = handler.Handler()
		query_article = 'this is an article'
		test_handler.get_recommended_articles(query_article=query_article)
		
if __name__ == '__main__':
    unittest.main()