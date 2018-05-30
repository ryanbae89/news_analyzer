# system import
import sys
import unittest

sys.path.append('/Users/paulwright/Dropbox/UW/2018_s_DATA515/Project/news_analyzer/news_analyzer/libraries')

# test imports
import pandas as pd
import configs
import nytimes_article_retriever as nytar



class TestNytimesArticleRetriever(unittest.TestCase):
    """ Usage: topic_modeling.py unit-test.

        python test_topic_modeling.py
    """
    @unittest.skip('need to fix api dependency')
    def setUp(self):
        self.all_topics = nytar.get_nytimes_data()
        self.some_topics = nytar.get_nytimes_data(['arts', 'automobiles', 'books'])
        self.all_topic_words = nytar.get_section_words(self.all_topics)
        self.some_topic_words = nytar.get_section_words(self.some_topics)
        self.get_all_topic_words = nytar.get_nytimes_topic_words()

    @unittest.skip('need to fix api dependency')
    def test_get_nytimes_data(self):
        """ Test to check getting NYtimes data.
        """
        # check return types
        self.assertTrue(isinstance(self.all_topics, pd.DataFrame))
        self.assertTrue(isinstance(self.some_topics, pd.DataFrame))
        # check length and shape of the return types
        self.assertTrue(len(self.all_topics) == len(configs.GUIDED_LDA_TOPICS))
        self.assertTrue(len(self.some_topics) == 3)
        self.assertTrue(self.all_topics.shape[1] == 2) # 2 columns of data
    
    @unittest.skip('need to fix api dependency')
    def test_get_section_words(self):
        """ Test to check getting NYTimes section words.
        """
        # check return type
        self.assertTrue(isinstance(self.all_topic_words, list))
        self.assertTrue(isinstance(self.some_topic_words, list))
        # check dimensions
        self.assertTrue(len(self.all_topic_words) == len(configs.GUIDED_LDA_TOPICS))
        self.assertTrue(len(self.some_topic_words) == 3)

    @unittest.skip('need to fix api dependency')
    def test_get_nytimes_topic_words(self):
        """ Test to check combined function for getting NYTimes data and extracting section words.
        """
        # check return type
        self.assertTrue(isinstance(self.get_all_topic_words, list))
        self.assertTrue(len(self.get_all_topic_words) == len(self.all_topic_words))
        #check each category has at least 10 words
        min_length = 10
        for i in range(len(self.get_all_topic_words)):
            min_length = min(min_length, len(self.get_all_topic_words[i]))
            #check each value is unique
            #print("testing lists")
            #print(len(self.get_all_topic_words[i]))
            #print(len(set(self.get_all_topic_words[i])))
            #print(self.get_all_topic_words[i])
            self.assertTrue(len(self.get_all_topic_words[i]) ==
                            len(set(self.get_all_topic_words[i])))
            #check first item is a category name
            self.assertTrue(self.get_all_topic_words[i][0] in configs.GUIDED_LDA_TOPICS)
        self.assertTrue(min_length >= 10)


if __name__ == '__main__':
    unittest.main()
