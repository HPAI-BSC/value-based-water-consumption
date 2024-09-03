import unittest
from pandas import DataFrame, read_csv
from analyzer import Analyzer
import random
class TestAnalyzer(unittest.TestCase):
    def test_analysis(self):
        df = read_csv('../src/analysis/all.csv')
        analyzer = Analyzer('../src/analysis/charts/')
        analyzer.analyze(df)

