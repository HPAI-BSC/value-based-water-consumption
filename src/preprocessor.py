from abc import ABC, abstractmethod

import pandas as pd
from pandas import DataFrame
from scipy.stats import chi2_contingency

from calendar import month_name


class Preprocessor(ABC):
    @abstractmethod
    def add_columns(self, filename: str, df: DataFrame):
        pass

    @abstractmethod
    def transform_categorical_to_ordinal(self, df: DataFrame):
        pass

    @abstractmethod
    def correlation(self, df: DataFrame):
        pass


class PreprocessorImpl(Preprocessor):
    def add_columns(self, filename: str, df: DataFrame):
        params = filename.split('|')
        if df.empty:
            return df
        else:
            new_df = df.copy()
            new_df['threshold_multiplier'] = float(params[1])
            new_df['num_agents'] = int(params[3])
            new_df['volume_per_block'] = int(params[4])
            new_df['block_multiplier'] = float(params[5])
            new_df['seed'] = int(params[6].split(')')[0])
            return new_df

    """Converts specific categorical columns into ordinalTreated columns: Density, income_type"""
    def transform_categorical_to_ordinal(self, df: DataFrame):
        if df.empty:
            return df
        else:
            new_df = df.copy()
            if 'density' in new_df.columns:
                new_df['density'].replace(['low density', 'medium density', 'high density'], [1, 2, 3], inplace=True)
            if 'income_type' in new_df.columns:
                new_df['income_type'].replace(['low', 'medium', 'high'], [1, 2, 3], inplace=True)
            if 'month' in new_df.columns:
                lower_ma = [m.lower() for m in month_name]
                new_df['month'] = new_df['month'].str.lower().map(lambda m: lower_ma.index(m)).astype('int64')
            return new_df

    """Checks dependency between two categorical variables (checking if they have a similar distribution).
        We use contingency since we do not know the actual distributions of categorical variables in df."""

    @staticmethod
    def chi_square_contingency(df: DataFrame, col1: str, col2: str):
        crosstab = pd.crosstab(df[col1], df[col2])
        return chi2_contingency(crosstab)

    def correlation(self, df: DataFrame, method='pearson'):
        new_df = df._get_numeric_data()  # Filters columns that are not numeric or boolean
        return new_df.corr(method=method)
