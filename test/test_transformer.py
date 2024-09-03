import unittest
from pandas._testing import assert_frame_equal, DataFrame

from preprocessor import PreprocessorImpl
from transformer import Transformer

class TestTransformer(unittest.TestCase):

    def test_transforms_everything(self):
        cols = ['month', 'agent_id', 'income', 'density', 'income_type']
        data = [['January', 1, 200.0, 'high density', 'low'],
                ['March', 2, 300.0, 'low density', 'high'],
                ['September', 5, 100.0, 'medium density', 'medium']]
        df = DataFrame(data, columns=cols)
        expected_data = [[1, 1, 200.0, 3, 1],
                         [3, 2, 300.0, 1, 3],
                         [9, 5, 100.0, 2, 2]]
        expected = DataFrame(expected_data, columns=cols)

        pr = PreprocessorImpl()
        trans = Transformer(pr)
        actual = trans.transform_data(df)
        assert_frame_equal(expected, actual)

    def test_transforms_what_is_actually_there(self):
        cols = ['agent_id', 'income', 'density', 'income_type']
        data = [[1, 200.0, 'high density', 'low'],
                [2, 300.0, 'low density', 'high'],
                [5, 100.0, 'medium density', 'medium']]
        df = DataFrame(data, columns=cols)
        expected_data = [[1, 200.0, 3, 1],
                         [2, 300.0, 1, 3],
                         [5, 100.0, 2, 2]]
        expected = DataFrame(expected_data, columns=cols)

        pr = PreprocessorImpl()
        trans = Transformer(pr)
        actual = trans.transform_data(df)

        assert_frame_equal(expected, actual)

    def test_transforms_nothing(self):
        cols = ['agent_id', 'income']
        data = [[1, 200.0], [2, 300.0], [5, 100.0]]
        df = DataFrame(data, columns=cols)
        expected = df.copy()

        pr = PreprocessorImpl()
        trans = Transformer(pr)
        actual = trans.transform_data(df)

        assert_frame_equal(expected, actual)

    def test_does_not_transform_what_it_cant(self):
        cols = ['month', 'agent_id', 'income', 'density', 'income_type', 'behaviour']
        data = [['January', 1, 200.0, 'high density', 'low', 'client'],
                ['March', 2, 300.0, 'low density', 'high', 'environmentalist'],
                ['September', 5, 100.0, 'medium density', 'medium', 'techno-solutionist']]
        df = DataFrame(data, columns=cols)
        expected_data = [[1, 1, 200.0, 3, 1, 'client'],
                         [3, 2, 300.0, 1, 3, 'environmentalist'],
                         [9, 5, 100.0, 2, 2, 'techno-solutionist']]
        expected = DataFrame(expected_data, columns=cols)

        pr = PreprocessorImpl()
        trans = Transformer(pr)
        actual = trans.transform_data(df)

        assert_frame_equal(expected, actual)