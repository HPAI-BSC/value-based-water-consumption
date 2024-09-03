import os
import pandas as pd
from preprocessor import PreprocessorImpl
from file_merger import FileMerger
# from feature_analysis import FeatureAnalysis
# from analyzer import Analyzer
# from transformer import Transformer


if __name__ == '__main__':
    # Once the experiments have been executed (Docker), we proceed to:
    # 1. Merge all resulting CSV files into a single one (all.csv)
    # 2. Preprocess it, transforming categorical features to numerical ones (transformed.csv)
    # 3. Compute correlation matrix and store a heatmap figure
    # 4. Proceed with analysis to compute LightGBM decision tree, storing two images (tree and feature relevance) for:
    #   4.1 All data
    #   4.2 Splitting data according to first-step in decision-making: cost-driven vs. consumption-driven
    #   4.3 Further split using density as criteria
    #   4.4 Last split, using income_type plus density as criteria
    # Resulting CSV will be gziped afterwards (shell script)

    # Setup all needed components
    preprocessor = PreprocessorImpl()
    file_merger = FileMerger(preprocessor)
    # transformer = Transformer(preprocessor)
    # feature_analyzer = FeatureAnalysis(preprocessor)
    # analyzer = Analyzer(os.getenv("RESULTS_PATH"))

    file_merger.merge_csvs("../results/weekly", "../results/") #os.getenv("RESULTS_PATH"))
    # transformed_df = transformer.transform_data(pd.read_csv(os.getenv("RESULTS_PATH")+'/all.csv'))
    # feature_analyzer.compute_correlation(transformed_df, path=os.getenv("RESULTS_PATH"))
    # analyzer.analyze(transformed_df)
    # analyzer.plot_reduction(transformed_df)
    # analyzer.consumption_charts_by_behaviour(transformed_df)
    # analyzer.consumption_charts_by_behaviour_and_density(transformed_df)
    # analyzer.consumption_charts_by_behaviour_density_and_income_type(transformed_df)
