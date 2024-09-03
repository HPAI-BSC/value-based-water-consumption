import unittest
import pandas as pd
from pandas import DataFrame
from preprocessor import PreprocessorImpl
from feature_analysis import FeatureAnalysis
class TestFeatureanalysis(unittest.TestCase):
    def test_compute_correlation(self):
        cols = ['month', 'agent_id', 'income', 'density', 'income_type', 'behaviour']
        data = [[1, 1, 200.0, 3, 1, 'client'],
                [3, 3, 300.0, 1, 3, 'environmentalist'],
                [9, 9, 100.0, 2, 2, 'techno-solutionist']]
        df = DataFrame(data, columns=cols)

        preprocessor = PreprocessorImpl()
        feature_analyzer = FeatureAnalysis(preprocessor)
        corr_df, too_correlated = feature_analyzer.compute_correlation(df, path='../src/analysis/charts')

        assert(('month', 'agent_id', 1.0) in too_correlated)
        assert(('density', 'income_type', -1.0) in too_correlated)
        assert('behaviour' not in corr_df.columns)