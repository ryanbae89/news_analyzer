"""
Unit Tests for WordCloudGenerator
"""
import sys
sys.path.append('../libraries')
import unittest
import WordCloudGenerator
import matplotlib.pyplot as plt
#from PIL import Image

# Define a class in which the tests will run
class WordCloudGeneratorTest(unittest.TestCase):
    """
    Unit TestCase Class for WordCloudGenerator
    """

    def test_smoke(self):
        """
        Basic Smoke Test

        Args:
            self (object): Reference to the class

        Returns:
            null
        """
        my_image = WordCloudGenerator.generate_wordcloud("random data")
        self.assertTrue(my_image is not None)


if __name__ == '__main__':
    unittest.main()