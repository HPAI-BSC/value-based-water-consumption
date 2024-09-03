from unittest import TestCase
import pandas as pd
from pandas import DataFrame
from pandas._testing import assert_frame_equal
import unittest
from preprocessor import PreprocessorImpl

class TestPreprocessor(TestCase):
    def test_add_columns_should_nothing_if_empty(self):
        df = DataFrame()
        filename = 'irrelevant|filename|because|df|is|empty'
        expected = df

        preprocessor = PreprocessorImpl()
        actual = preprocessor.add_columns(filename, df)

        assert_frame_equal(expected, actual)

    def test_add_columns_should_add_threshold_multiplier(self):
        data = [[1, 2], [3, 4]]
        df = DataFrame(data, columns=['a', 'b'])
        filename = f'monthly (0|0.5|0|1000|6|1.1|42).csv'
        expected_data = [[1, 2, 0.5, 1000, 6, 1.1, 42], [3, 4, 0.5, 1000, 6, 1.1, 42]]
        expected = DataFrame(expected_data, columns=['a', 'b', 'threshold_multiplier',
                                                     'num_agents', 'volume_per_block',
                                                     'block_multiplier', 'seed'])

        preprocessor = PreprocessorImpl()
        actual = preprocessor.add_columns(filename, df)

        assert_frame_equal(expected, actual)

    def test_transform_categorical_into_ordinal(self):
        cols = ['members', 'income_type', 'density', 'income', 'behaviour', 'month']
        data = [[1, 'high', 'medium density', 3, 'client', 'January'],
                [2, 'medium', 'low density', 1, 'techno-solutionist', 'April'],
                [3, 'low', 'high density', 5, 'environmentalist', 'September']]
        df = DataFrame(data, columns=cols)
        expected_data = [[1, 3, 2, 3, 'client', 1],
                         [2, 2, 1, 1, 'techno-solutionist', 4],
                         [3, 1, 3, 5, 'environmentalist', 9]]
        expected = DataFrame(expected_data, columns=cols)
        preprocessor = PreprocessorImpl()
        actual = preprocessor.transform_categorical_to_ordinal(df)
        assert_frame_equal(expected, actual)

    def test_transform_categorical_into_ordinal_that_are_present(self):
        cols = ['members', 'income_type', 'density', 'income', 'behaviour']
        data = [[1, 'high', 'medium density', 3, 'client'],
                [2, 'medium', 'low density', 1, 'techno-solutionist'],
                [3, 'low', 'high density', 5, 'environmentalist']]
        df = DataFrame(data, columns=cols)
        expected_data = [[1, 3, 2, 3, 'client'],
                         [2, 2, 1, 1, 'techno-solutionist'],
                         [3, 1, 3, 5, 'environmentalist']]
        expected = DataFrame(expected_data, columns=cols)
        preprocessor = PreprocessorImpl()
        actual = preprocessor.transform_categorical_to_ordinal(df)
        assert_frame_equal(expected, actual)

    def test_transform_categorical_into_ordinal_does_nothing_empty(self):
        df = DataFrame()
        expected = df
        preprocessor = PreprocessorImpl()
        actual = preprocessor.transform_categorical_to_ordinal(df)
        assert_frame_equal(expected, actual)


    def test_correlation_for_mixed_datatypes(self):
        data = [[3, 2, 1, False, 'Client'],
                [1, 2, 3, True, 'Environmentalist'],
                [-1, -1, -1, False, 'Comitted']]
        df = DataFrame(data, columns=['a', 'b', 'c', 'd', 'Profile'])
        expected = DataFrame(columns=['a','b','c','d'])
        preprocessor = PreprocessorImpl()
        actual = preprocessor.correlation(df)
        cols = actual.columns
        for col in cols:
            print(actual[col])
        assert(expected.columns, actual.columns)
