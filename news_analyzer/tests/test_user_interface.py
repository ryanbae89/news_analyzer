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
import user_interface as ui


class TestUserInterface(unittest.TestCase):
    """ Usage: unit-test.

        python test_user_interface.py
    """


    def test_make_dash_table(self):
        """ Test to check getting NYtimes data.
        """
        output = ui.make_dash_table(pd.DataFrame(np.asarray([[1,2],[3,4],[5,6]])))
        # check return types
        self.assertTrue(isinstance(output, list))
        # check length and shape of the return types
        self.assertTrue(len(output) == 3)


if __name__ == '__main__':
    unittest.main()
