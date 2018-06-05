""" Module for testing text_processing library.
"""
import unittest
import sys

from nltk.corpus import stopwords
import numpy as np
import pandas as pd

sys.path.append('../libraries')
# pylint: disable=wrong-import-position
import text_processing as tpp # noqa

STOPWORDS = list(set(stopwords.words('english'))) + tpp.EXTRA_STOPWORDS


class ArticleCleaningTest(unittest.TestCase):
    """ These tests will test the functions in the module
        clean_article and transform article.
    """

    def setUp(self):
        """ Test set up.
        """
        article_1 = "This should be an easy article to parse.\
                            \n It has 2 sentences and a newline."
        article_2 = "This is a single sentence and has a stopword."
        article_3 = "SingleWord"
        article_4 = "L"
        article_5 = " "
        article_6 = ""
        article_7 = "!@#$%^&"
        self.test_articles = [article_1, article_2, article_3,
                              article_4, article_5, article_6,
                              article_7]

    def test_clean_article(self):
        """ Input for tpp.clean_article function
            should be a string (single or multiline)
            and output should be word list in the same order
            as input.

            Tests:
            * No punctuation.
            * All lower case.
            * Stopwords removed (ensure using extended list).
            * Doesn't fail with only 1 word.
            * Returns error if no text to return or inputted.
        """
        for i, test_article in enumerate(self.test_articles):
            with self.subTest(test_article=i):
                cleaned_list = tpp.clean_article(test_article)
                # Test for no punctuation
                self.assertTrue(is_alphabetic_only(cleaned_list))
                # Test for no lowercase
                self.assertTrue(is_lowercase(cleaned_list))
                # Test for no stopwords
                self.assertFalse(has_stopwords(cleaned_list))

    def test_transform_article(self):
        """ Input for tpp.transform_article function is a single article.

            Same tests as in test_clean_article.
        """
        for i, test_article in enumerate(self.test_articles):
            with self.subTest(test_article=i):
                cleaned_article = tpp.transform_article(test_article)
                # Test for no punctuation
                self.assertTrue(is_alphabetic_only(cleaned_article))
                # Test for no lowercase
                self.assertTrue(is_lowercase(cleaned_article))
                # Test for no stopwords
                self.assertFalse(has_stopwords(cleaned_article))
                # Test for equal or less number of words.


class ArticlePreprocessorTest(unittest.TestCase):
    """ These will test the ArticlePreprocessor() class.
    """

    def setUp(self):
        """ Test set up.
        """
        self.processor = tpp.ArticlePreprocessor()
        article_1 = "This should be an easy article to parse.\
                            \n It has 2 sentences and a newline."
        article_2 = "This is a single sentence and has a stopword."
        article_3 = "SingleWord"
        article_4 = "L"
        article_5 = " "
        article_6 = ""
        article_7 = "!@#$%^&"
        self.articles_should_pass = [article_1, article_2, article_3]
        self.articles_should_fail = [article_4, article_5,
            article_6, article_7]
        self.test_articles = self.articles_should_pass + \
            self.articles_should_fail

    def test_fit_on_good_articles(self):
        """ Tests whether fit function returns any errors using
            our good test articles.
        """

        for test_article in self.articles_should_pass:
            with self.subTest(test_article=test_article):
                try:
                    self.processor.fit(test_article)
                except Exception:
                    self.fail('Error thrown when test_article fit in \
                                preprocessor')

        # Test if fit function works on whole corpus (list of articles)
        # Only a subset of the corpus needs to pass for fit to work.
        try:
            self.processor.fit(self.articles_should_pass +
                        self.articles_should_fail)
        except Exception:
            self.fail('Error thrown when corpus fit in \
                        preprocessor')

    def test_fit_on_bad_articles(self):
        """ Tests if we get errors when passing in bad (un-dtm-able)
            articles.
        """
        for test_article in self.articles_should_fail:
            with self.subTest(test_article=test_article):
                try:
                    self.processor.fit(test_article)
                    self.fail("Preprocessor should have \
                        returned a ValueError.")
                except ValueError:
                    pass

    def test_transform(self):
        """ Tests to ensure (1) transforms work on multiple
            types of inputs once processor is fit, (2) subsequent
            fits override previous ones, (3) the document-term-matrix
            is in the correct format and has the correct words.

            This test will not test the functionality of scikit-learn's
            CountVectorizer class within ArticlePreprocessor (max features,
            min_df, etc.).

        """
        self.processor.fit(self.test_articles)
        for test_article in self.test_articles:
            with self.subTest(test_article=test_article):
                dtm = self.processor.transform(test_article)
                # Test if dtm is pandas dataframe
                self.assertTrue(isinstance(dtm, pd.DataFrame))
                # Test shape of dtm
                self.assertTrue(dtm.shape[1] == self.processor.dtm.shape[1])
                self.assertTrue(dtm.shape[0] == 1)

        # Test for entire corpus
        dtm = self.processor.transform(self.test_articles)
        cols = dtm.columns
        # Test shape
        self.assertTrue(dtm.shape[0] == len(self.test_articles))
        # Test against query articles
        query_dtm_1 = self.processor.transform("text outside of test cases")
        # Test if columns are the same
        self.assertTrue(dtm.shape[1] == query_dtm_1.shape[1])
        self.assertTrue(query_dtm_1.columns.isin(cols).all())
        query_dtm_2 = self.processor.transform("text inside of test cases \
            like 'newline'")
        self.assertTrue(dtm.shape[1] == query_dtm_2.shape[1])
        self.assertTrue(query_dtm_2.columns.isin(cols).all())

    def test_get_vocab(self):
        """ Test for get_vocab() method.
        """
        self.processor.fit(self.test_articles)
        dtm = self.processor.dtm
        cols = dtm.columns
        num_cols = len(dtm.columns)

        vocab = self.processor.get_vocab()
        # Ensure size is the same
        self.assertTrue(len(vocab) == num_cols)
        # Ensure words are the same
        self.assertTrue(cols.isin(vocab).all())


def has_stopwords(string):
    """Internal function for checking if string is lowercase.
    Args:
    string (str): string to check

    Returns:
    True/False: Whether or not s contains any numbers of special characters
    """
    if hasattr(string, "split"):
        s_list = pd.Series(string.split())
    else:
        s_list = pd.Series(string)
    return s_list.isin(STOPWORDS).any()


def is_lowercase(string):
    """Internal function for checking if string is lowercase.
    Args:
    string (str): string to check

    Returns:
    True/False: Whether or not s contains any numbers of special characters
    """
    if hasattr(string, "split"):
        s_list = string.split()
    else:
        s_list = string
    return np.array([substr.islower() for substr in s_list]).all()


def is_alphabetic_only(string):
    """Internal function for checking if string is alphabetic.
    Args:
    s (str): string to check

    Returns:
    True/False: Whether or not s contains any numbers of special characters
    """
    if hasattr(string, "split"):
        s_list = string.split()
    else:
        s_list = string
    return np.array([substr.isalpha() for substr in s_list]).all()


if __name__ == '__main__':
    unittest.main()
