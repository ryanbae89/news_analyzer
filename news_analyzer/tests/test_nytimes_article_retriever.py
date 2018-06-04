"""
This module performs unit tests on the nytimes article retriever class.
"""
# system import
import sys
import unittest
import pandas as pd

sys.path.append('../libraries')
sys.path.append('news_analyzer/libraries')


# test imports
# pylint: disable=wrong-import-position
import configs # noqa
import nytimes_article_retriever as nytar # noqa


class TestNytimesArticleRetriever(unittest.TestCase):
    """ Usage: unit-test.

        python test_topic_modeling.py
    """
    def setUp(self):
        # makes calls to API
        self.all_topics = nytar.get_nytimes_data()
        # makes calls to API
        self.some_topics = nytar.get_nytimes_data(['arts',
                                                   'automobiles',
                                                   'books'])
        self.all_topic_words = nytar.get_section_words(self.all_topics)
        self.some_topic_words = nytar.get_section_words(self.some_topics)
        self.get_all_topic_words = nytar.get_nytimes_topic_words()
        self.aggregated_data = nytar.aggregate_data()

    def test_get_nytimes_data(self):
        """ Test to check getting NYtimes data.
        """
        # check return types
        self.assertTrue(isinstance(self.all_topics, pd.DataFrame))
        self.assertTrue(isinstance(self.some_topics, pd.DataFrame))
        # check length and shape of the return types
        # print("all topics:")
        # print(len(self.all_topics))
        # self.assertTrue(len(self.all_topics) ==
        #    len(configs.GUIDED_LDA_TOPICS))
        # self.assertTrue(len(self.some_topics) == 3)
        # self.assertTrue(self.all_topics.shape[1] == 2)  # 2 columns of data

    def test_aggregate_data(self):
        """ Test to check process for aggregating data.
        """
        # check return types
        self.assertTrue(isinstance(self.aggregated_data, pd.DataFrame))
        # check length and shape of the return types
        # print(len(self.aggregated_data))
        # print(len(configs.GUIDED_LDA_TOPICS))
        # self.assertTrue(len(self.aggregated_data) ==
        #                len(configs.GUIDED_LDA_TOPICS))

    def test_get_section_words(self):
        """ Test to check getting NYTimes section words.
        """
        # check return type
        self.assertTrue(isinstance(self.all_topic_words, list))
        self.assertTrue(isinstance(self.some_topic_words, list))

        # check dimensions
        # self.assertTrue(len(self.all_topic_words) ==
        #                len(configs.GUIDED_LDA_TOPICS))
        # self.assertTrue(len(self.some_topic_words) == 3)

    def test_get_nytimes_topic_words(self):
        """ Test to check combined function for getting NYTimes data and
        extracting section words.
        """
        # check return type
        self.assertTrue(isinstance(self.get_all_topic_words, list))
        # print("all topic words")
        # print(len(self.get_all_topic_words))
        # print(len(self.all_topic_words))
        # self.assertTrue(len(self.get_all_topic_words) ==
        #                len(self.all_topic_words))

        # check each category has at least 10 words
        min_length = 10
        for i in range(len(self.get_all_topic_words)):
            min_length = min(min_length, len(self.get_all_topic_words[i]))
            # check each value is unique
            self.assertTrue(len(self.get_all_topic_words[i]) ==
                            len(set(self.get_all_topic_words[i])))
            # check first item is a category name
            self.assertTrue(self.get_all_topic_words[i][0] in
                            configs.GUIDED_LDA_TOPICS)
        print("min_length:", min_length)
        self.assertTrue(min_length >= 10)


if __name__ == '__main__':
    unittest.main()
