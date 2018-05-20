# test imports
import numpy as np 
import pandas as pd
import guidedlda
import unittest
import topic_modeling

class TestTopicModeling(unittest.TestCase):
	""" Usage: topic_modeling.py
	"""
	def test_check(self):
		""" Test to check if the test runs.
		"""
		self.assertTrue(1 > 0)

	def test_get_vocab(self):
		""" Test to check get_vocab function.
		"""
		# pandas dataframe test case
		self.test_dtm = pd.read_pickle('test_dtm.pkl')
		vocab, word2id = topic_modeling.get_vocab(self.test_dtm)
		# check return types
		self.assertTrue(type(vocab) == list)
		self.assertTrue(type(word2id) == dict)
		# check length and shape of the return types
		self.assertTrue(len(vocab) == self.test_dtm.shape[1])
		self.assertTrue(len(word2id) == self.test_dtm.shape[1])
		# check content
		self.assertTrue(np.all(vocab == self.test_dtm.columns))
		self.assertTrue(list(word2id.keys()) == vocab)

	def test_TopicModler(self):
		""" Tests for the TopicModeler class.
		"""
		# test for non-integer inputs
		with self.assertRaises(Exception) as context:
			topic_modeling.TopicModeler(1.0, 0, 0)
		self.assertTrue('Inputs to TopicModeler must be non-negative integers!' \
			in str(context.exception))
		# test for integer inputs less than 0
		with self.assertRaises(Exception) as context:
			topic_modeling.TopicModeler(1, -1, 0)
		self.assertTrue('Inputs to TopicModeler must be non-negative integers!' \
			in str(context.exception))
		# test fit function (unguided case)
		self.test_dtm = pd.read_pickle('test_dtm.pkl')
		test_modeler = topic_modeling.TopicModeler(n_topics=10, n_iter=100, 
			random_state=0, refresh=20)
		test_model = test_modeler.fit(self.test_dtm)
		self.assertTrue(type(test_model) == guidedlda.guidedlda.GuidedLDA)
		# test fit function (guided case)
		test_modeler = topic_modeling.TopicModeler(n_topics=10, n_iter=100, 
			random_state=0, refresh=20)
		test_model = test_modeler.fit(self.test_dtm)
		self.assertTrue(type(test_model) == guidedlda.guidedlda.GuidedLDA)

SUITE = unittest.TestLoader().loadTestsFromTestCase(TestTopicModeling)
_ = unittest.TextTestRunner().run(SUITE)