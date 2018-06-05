"""
This module performs unit tests on the user interface class.
"""
# system import
import sys
import unittest
import numpy as np
import pandas as pd

sys.path.append('../libraries')
sys.path.append('news_analyzer/libraries')


# test imports
# pylint: disable=wrong-import-position
import user_interface as ui # noqa


class TestUserInterface(unittest.TestCase):
    """ Usage: unit-test. Mostly smoke tests as the handler class does most
        of the validation and the interface class converts to dash compatible
        code.

        python test_user_interface.py
    """

    def test_make_dash_table(self):
        """ Test to check getting NYtimes data.
        """
        output = ui.make_dash_table(pd.DataFrame(np.asarray([[1, 2],
                                                             [3, 4],
                                                             [5, 6]])))
        # check return types
        self.assertTrue(isinstance(output, list))
        # check length and shape of the return types
        self.assertTrue(len(output) == 3)

    def test_recommended_articles(self):
        """
        test the update_recommended_articles function
        """
        output = ui.update_recommended_articles(4, "test query article")
        self.assertIsNotNone(output)

    def test_sentiment_information(self):
        """
        test the update_sentiment_information function
        """
        output = ui.update_sentiment_information(4, "test query article")
        self.assertIsNotNone(output)

    def test_update_top_topics(self):
        """
        test the update_top_topics function
        """
        output = ui.update_top_topics(4, "test query article")
        self.assertIsNotNone(output)

    def test_update_word_cloud_image(self):
        """
        test the update_word_cloud_image function
        """
        output = ui.update_word_cloud_image(4, "test query article")
        self.assertIsNotNone(output)


if __name__ == '__main__':
    unittest.main()
