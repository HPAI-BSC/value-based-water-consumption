import os
import tempfile
from unittest import TestCase

import pandas as pd
from pandas import DataFrame
from pandas._testing import assert_frame_equal

from file_merger import FileMerger
from preprocessor import Preprocessor


class PreprocessorMock(Preprocessor):
    def add_columns(self, filename: str, df: DataFrame):
        new_df = df.copy()
        new_df['mock'] = True
        return new_df


class TestFileMerger(TestCase):

    def test_file_merger(self):
        with tempfile.TemporaryDirectory() as dirname:
            print(dirname)
            mock_preprocessor = PreprocessorMock()
            file_merger = FileMerger(mock_preprocessor)

            file_merger.merge_csvs(os.path.join('..', 'resources', 'sample_csvs'), dirname)
            expected = pd.read_csv(os.path.join('..', 'resources', 'sample_joined', 'all.csv'))
            actual = pd.read_csv(os.path.join(dirname, 'all.csv'))

            assert_frame_equal(expected.sort_values(by='agent', ignore_index=True),
                               actual.sort_values(by='agent', ignore_index=True), check_like=True)
