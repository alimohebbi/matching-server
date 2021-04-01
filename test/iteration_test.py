import unittest

import pandas as pd

pd.DataFrame({'x': [1, 2, 3], 'y': [3, 4, 5]})



class TestBasic(unittest.TestCase):
    def test_iter_row(self):
        df = pd.DataFrame({'x': [1, 2, 3], 'y': [3, 4, 5], 'i':[100,1002,120]})
        df.set_index('i', inplace=True)
        for k, i in df.iterrows():
            df.loc[k, 'x'] = int(k)

