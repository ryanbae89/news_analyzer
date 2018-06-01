"""
This module conducts unittest on the topic_modeling.py module.
"""
# standard imports
import sys
import unittest
import pickle
import guidedlda
import numpy as np
import pandas as pd
# test import
sys.path.append('news_analyzer/libraries')
import topic_modeling


class TestTopicModeling(unittest.TestCase):
    """ Usage: topic_modeling.py unit-test.
        python test_topic_modeling.py
    """
    def setUp(self):
        self.test_dtm = pd.read_pickle(
            'news_analyzer/tests/test_resources/test_dtm.pkl')
        topics_path = "news_analyzer/tests/test_resources/test_topics_raw.pkl"
        with open(topics_path, "rb") as file_handle:
            self.raw_topics = pickle.load(file_handle)
        self.bad_topics = ['national', 'nyregion', 'obituaries']
        self.vocab, self.word2id = topic_modeling.get_vocab(self.test_dtm)
        self.clean_topics = topic_modeling.clean_topics(
            self.raw_topics, self.word2id, self.bad_topics)
        self.seed_topics = topic_modeling.get_seed_topics(
            self.clean_topics, self.word2id)

    def test_get_vocab(self):
        """ Test to check get_vocab function.
        """
        # check return types
        self.assertTrue(isinstance(self.vocab, list))
        self.assertTrue(isinstance(self.word2id, dict))
        # check length and shape of the return types
        self.assertTrue(len(self.vocab) == self.test_dtm.shape[1])
        self.assertTrue(len(self.word2id) == self.test_dtm.shape[1])
        # check content
        self.assertTrue(np.all(self.vocab == self.test_dtm.columns))
        self.assertTrue(set(self.word2id.keys()) == set(self.vocab))

    def test_clean_topics(self):
        """ Tests the clean_topics method.
        """
        # check return type
        self.assertTrue(isinstance(self.clean_topics, list))

    def test_get_seed_topics(self):
        """ Tests the get_seed_topics method.
        """
        # check return type
        self.assertTrue(isinstance(self.seed_topics, dict))

    def test_topic_modeler(self):
        """ Tests for the TopicModeler class.
        """
        # test for non-integer inputs
        with self.assertRaises(Exception) as context:
            topic_modeling.TopicModeler(1.0, 0, 0)
        self.assertTrue(
            'Inputs to TopicModeler must be non-negative integers!'
            in str(context.exception))
        # test for integer inputs less than 0
        with self.assertRaises(Exception) as context:
            topic_modeling.TopicModeler(1, -1, 0)
        self.assertTrue(
            'Inputs to TopicModeler must be non-negative integers!'
            in str(context.exception))
        # non-valid dtm input
        test_modeler = topic_modeling.TopicModeler(n_topics=5)
        with self.assertRaises(Exception) as context:
            bad_dtm = [0, 1, 0]
            test_modeler.fit(bad_dtm)
        self.assertTrue(
            'Please input a valid pandas dataframe or numpy array for dtm!'
            in str(context.exception))
        # check return type (unguided case)
        test_modeler = topic_modeling.TopicModeler(
            n_topics=10, n_iter=100, random_state=0, refresh=20)
        test_model = test_modeler.fit(self.test_dtm)
        self.assertTrue(isinstance(test_model, guidedlda.guidedlda.GuidedLDA))
        # non-valid inputs (guided case)
        test_modeler = topic_modeling.TopicModeler(
            n_topics=20, n_iter=100, random_state=0, refresh=20)
        with self.assertRaises(Exception) as context:
            test_model = test_modeler.fit(
                dtm=self.test_dtm, seed_topics=[], seed_confidence=0.5)
        self.assertTrue('Please enter a dictionary for seed_topics.'
                        in str(context.exception))
        with self.assertRaises(Exception) as context:
            test_model = test_modeler.fit(
                dtm=self.test_dtm,
                seed_topics=self.seed_topics,
                seed_confidence=1)
        self.assertTrue('Please enter a float for seed_confidence.'
                        in str(context.exception))
        test_modeler = topic_modeling.TopicModeler(
            n_topics=10, n_iter=100, random_state=0, refresh=20)
        with self.assertRaises(Exception) as context:
            test_model = test_modeler.fit(
                dtm=self.test_dtm,
                seed_topics=self.seed_topics,
                seed_confidence=0.5)
        self.assertTrue(
            'n_topics must be greater than number of seed topics!'
            in str(context.exception))
        # check return type (guided case)
        test_modeler = topic_modeling.TopicModeler(
            n_topics=len(self.seed_topics),
            n_iter=100,
            random_state=0,
            refresh=20)
        test_model = test_modeler.fit(
            dtm=self.test_dtm,
            seed_topics=self.seed_topics,
            seed_confidence=0.5)
        self.assertTrue(isinstance(test_model, guidedlda.guidedlda.GuidedLDA))

    def test_topic_modeler_gridsearch(self):
        """ Tests for the TopicModelerGridSearch class.
        """
        # check n_topics_list input
        with self.assertRaises(Exception) as context:
            test_gridsearch = topic_modeling.TopicModelerGridSearch(
                '2', 100, 0, 20)
        self.assertTrue('You must enter a valid list of n_topics.'
                        in str(context.exception))
        # check return type
        test_gridsearch = topic_modeling.TopicModelerGridSearch(
            n_topics_list=[5, 10], n_iter=60, random_state=0, refresh=20)
        test_gridsearch.gridsearch(self.test_dtm)
        self.assertTrue(isinstance(
            test_gridsearch.model, guidedlda.guidedlda.GuidedLDA))
        self.assertTrue(isinstance(test_gridsearch.loglikelihoods, list))
        self.assertTrue(isinstance(test_gridsearch.n_topics_opt, int))
        # check output lengths
        self.assertTrue(len(test_gridsearch.n_topics_list)
                        == len(test_gridsearch.loglikelihoods))


if __name__ == '__main__':
    unittest.main()
