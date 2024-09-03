from pandas import DataFrame
from preprocessor import Preprocessor

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import logging
import json

class FeatureAnalysis(object):
    def __init__(self, preprocessor: Preprocessor):
        self.preprocessor = preprocessor


    """Creates a PNG file containing heatmap of given correlation matrix (triangle)."""
    def __create_correlation_heatmap(self, corr_df: DataFrame, title: str='Correlation heatmap',
                                     filename: str='Correlation_heatmap', path_figure: str='.'):
        # mask_ut = np.triu(np.ones(corr_df.shape), 1).astype(bool) # We keep the diagonal

        hmap = sns.heatmap(corr_df.abs(), xticklabels=1, yticklabels=1, cmap='Spectral') # , mask=mask_ut)
        plt.title(title)
        hmap.figure.tight_layout()
        hmap.figure.savefig(path_figure+"/"+filename +".png",
                            format='png',
                            dpi=250)
        plt.close()

    """Computes the correlation matrix of df. Ignores columns and rows with NaN values."""
    def __get_correlation_matrix(self, df: DataFrame, method='pearson'):
        corr_df = self.preprocessor.correlation(df, method)
        corr_df.dropna(axis=0, how='all', inplace=True) # Removes rows with all NaN
        corr_df.dropna(axis=1, how='all', inplace=True) # Removes columns with all NaN (e.g., mock)
        return corr_df#corr_df.where(np.tril(np.ones(corr_df.shape)).astype(bool))


    """Returns the correlation matrix (lower triangle) and a list of triples of those pair of features, along their
    correlation value, that have overpassed given threshold. Stores a figure of the correlation matrix as a heatmap."""
    def compute_correlation(self, df: DataFrame, threshold: float=0.7, path: str='./'):
        res = self.__get_correlation_matrix(df) # We are using the whole matrix, not the triangle
        too_correlated = list()
        for index, row in res.iterrows():
            for i, value in row.items():
                if i == index: # Since it is a triangle, next values are going to be NaN
                    break
                elif (threshold <= abs(value)): # We want to check threshold for positive/negative correlation
                    too_correlated.append((i, index, value))

        self.__create_correlation_heatmap(res, path_figure=path)
        res.to_csv(path+"/correlation_matrix.csv")
        with open(path+"/too_correlated.json", 'w') as fp:
            json.dump(too_correlated, fp)
        return res, too_correlated

