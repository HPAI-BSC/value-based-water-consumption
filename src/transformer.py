from pandas import DataFrame
from preprocessor import Preprocessor
"""This class takes care of transforming the columns of CSV file as required for next steps. 
e.g., categorical to numerical"""
class Transformer(object):
    def __init__(self, preprocessor: Preprocessor):
        self.preprocessor = preprocessor

    def transform_data(self, df: DataFrame):
        new_df = df.copy()
        new_df = self.preprocessor.transform_categorical_to_ordinal(new_df)
        return new_df
