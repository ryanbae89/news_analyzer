"""
Unit Tests for WordCloudGenerator
"""
import unittest
import sys
sys.path.append('news-article-nlp/libraries')

import word_cloud_generator


# Define a class in which the tests will run
class WordCloudGeneratorTest(unittest.TestCase):
    """
    Unit TestCase Class for WordCloudGenerator
    """

    def test_short_article(self):
        """
        Basic Smoke Test

        Args:
            self (object): Reference to the class

        Returns:
            null
        """
        my_image = word_cloud_generator.generate_wordcloud("random data")
        self.assertTrue(my_image is not None)

    def test_longer_article(self):
        """
        Basic Smoke Test

        Args:
            self (object): Reference to the class

        Returns:
            null
        """
        my_image = word_cloud_generator.generate_wordcloud(
            "(CNN)ABC's decision to cancel Roseanne Barr's eponymous show "
            "following a racist comment she made about former Obama "
            "administration official Valerie Jarrett on Twitter was "
            "shocking for two reasons. First, because it amounted to a TV "
            "network drawing a moral line in the sand -- insisting that no "
            "amount of money or ratings gave Roseanne the right to express "
            "views that ABC described in a statement as abhorrent, repugnant "
            "and inconsistent with our values.")
        self.assertTrue(my_image is not None)


if __name__ == '__main__':
    unittest.main()
