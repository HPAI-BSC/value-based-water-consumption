import pandas as pd
from pandas import DataFrame
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import logging
import datetime, time
import os
import graphviz, pydot
import numpy as np
from calendar import month_name, monthrange


from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

warnings.simplefilter(action='ignore', category=FutureWarning)

default_hyper_params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': ['l2', 'auc'],
    'learning_rate': 0.5,
    "num_leaves": 256,
    "max_bin": 600,
}


class Analyzer(object):
    def __init__(self, path):
        self.path = path

    def plot_consumptions(self, consumptions, title: str=''):
        # consumptions = [{'series_name_key': {'mean': [], 'std': []}}
        x = month_name[1:]
        for key, value in consumptions.items():
            mean = value['mean']
            std = value['std']
            plt.plot(x, mean, label=key)
            plt.fill_between(x, mean - std, mean + std, alpha=0.1)
        plt.legend(fontsize=12)
        plt.title('Monthly consumption (m3) ' + title, fontsize=12)
        plt.xticks(rotation=90, fontsize=10)
        plt.yticks(fontsize=10)
        plt.savefig(os.path.join(self.path, 'consumptions-' + title + '.png'))
        plt.close()


    def plot_reduction(self, df: DataFrame, filename:str=''):
        x = month_name[1:]
        df_transformed = df.copy()
        df_transformed['avg_daily_consumption_per_capita'] = df_transformed.apply(lambda row: row['consumption_per_capita (m3)'] / monthrange(2015, row['month'])[1], axis=1)
        df_by_month = df_transformed[['month', 'consumption (m3)', 'consumption_per_capita (m3)', 'avg_daily_consumption_per_capita', 'devices_changed_acc', 'practices_changed_acc']]
        mean = df_by_month.groupby(['month']).mean()
        std = df_by_month.groupby(['month']).std()
        df_aux = df_by_month.groupby(['month']).sum().sort_values(by=['month'])
        init = df_by_month.groupby(['month']).sum().sort_values(by=['month']).iloc[0]['avg_daily_consumption_per_capita']
        df_aux['perc_reduction_avg_daily_consumption_per_capita'] = (df_aux['avg_daily_consumption_per_capita'] - init) / init

        values1 = df_aux['devices_changed_acc']
        values2 = df_aux['practices_changed_acc']
        fig, ax = plt.subplots()

        ax.bar(month_name[1:], values2, label='Practices', alpha=0.2)
        ax.bar(month_name[1:], values1, label='Devices', bottom=values2, alpha=0.2)
        ax.set_ylabel('Number changed/adopted (accum.)', fontsize=12)
        for bar in ax.patches:
            ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() / 2 + bar.get_y(),
                round(bar.get_height()), ha = 'center',
                color = 'w', weight = 'bold', size = 10)

        plt.xticks(rotation=90, fontsize=10)
        plt.yticks(fontsize=10)
        plt.legend(fontsize=12)

        ax2 = plt.twinx()
        ax2.plot(month_name[1:], df_aux['perc_reduction_avg_daily_consumption_per_capita'])
        ax2.set_ylabel('% consumption reduction', fontsize=12)

        plt.savefig(os.path.join(self.path, 'perc-reduction-avg-daily-per-capita-and-saving-actions-' + filename + '.png'))
        plt.close()


    def plot_feature_importance(self, feature_imp:DataFrame, filename: str, max_value: float, min_value:float, num=20, fig_size=(40, 20)):
        #feature_imp = pd.DataFrame({'Value': model.feature_importance(importance_type='gain'), 'Feature': X.columns})
        plt.figure(figsize=fig_size)
        sns.set(font_scale=5)

        g = sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False)[0:num])
        plt.xlim(min_value, max_value)
        plt.title('LightGBM Features (avg over folds) ' + filename)
        plt.tight_layout()
        plt.savefig(os.path.join(self.path, 'lgbm_importances-' + filename + '.png'))
        plt.close()
        #plt.show()

    def print_decision_tree(self, model, X, filename):
        graph = lgb.create_tree_digraph(model, tree_index=0)
        graph.render(self.path + 'decision_tree-' + filename, format='png', cleanup=True)

    def experiment(self, hyper_params, X_train, X_test, y_train, y_test):
        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_test = lgb.Dataset(X_test, y_test, reference=lgb_train)
        gbm = lgb.train(hyper_params, train_set=lgb_train, valid_sets=[lgb_test], num_boost_round=10)
        y_fit = gbm.predict(X_train)
        y_pred = gbm.predict(X_test)
        mse_fit = mean_squared_error(y_train, y_fit)
        rmse_fit = mse_fit ** 0.5
        mse_test = mean_squared_error(y_test, y_pred)
        rmse_test = mse_test ** 0.5
        #print("MSE_fit: %.2f" % mse_fit)
        #print("RMSE_fit: %.2f" % rmse_fit)
        #print("MSE_test: %.2f" % mse_test)
        #print("RMSE_test: %.2f" % rmse_test)
        return rmse_fit, rmse_test, gbm

    def find_num_leaves(self, X_train, X_test, y_train, y_test):
        num_leaves = [8, 16, 32, 50, 64, 70, 80, 90, 100, 110, 128, 150, 200, 256, 300, 350, 400, 450, 512, 1024]

        results = pd.DataFrame()
        for n in num_leaves:
            hyper_params = default_hyper_params.copy()
            hyper_params['num_leaves'] = n
            rmse_fit, rmse_test, _ = self.experiment(hyper_params)
            results[n] = {'rmse_fit': rmse_fit, 'rmse_test': rmse_test}
        print(results)
        results.transpose().plot()
        plt.show()

    def find_max_bins(self, X_train, X_test, y_train, y_test):
        max_bins = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]

        results = pd.DataFrame()
        for n in max_bins:
            hyper_params = default_hyper_params.copy()
            hyper_params['max_bins'] = n
            rmse_fit, rmse_test, _ = self.experiment(hyper_params, X_train, X_test, y_train, y_test)
            results[n] = {'rmse_fit': rmse_fit, 'rmse_test': rmse_test}
        print(results)
        results.transpose().plot()
        plt.show()

    def find_learning_rate(self):
        learning_rate = [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.05, 0.005, 0.0005, 0.00005]

        results = pd.DataFrame()
        for n in learning_rate:
            hyper_params = default_hyper_params.copy()
            hyper_params['learning_rate'] = n
            rmse_fit, rmse_test, _ = self.experiment(hyper_params)
            results[n] = {'rmse_fit': rmse_fit, 'rmse_test': rmse_test}
        print(results)
        results.transpose().plot()
        plt.show()


    def perform_gbm_analysis(self, df: DataFrame, filename: str):
        y = df['consumption (m3)']
        X = df.drop(['consumption (m3)'], axis=1)

        for c in X.columns:
            col_type = X[c].dtype
            if col_type == 'object' or col_type.name == 'category':
                X[c] = X[c].astype('category')

        X.info()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)

        # find_num_leaves()
        # find_max_bins()
        # find_learning_rate()
        _, _, gbm = self.experiment(default_hyper_params, X_train, X_test, y_train, y_test)
        self.print_decision_tree(gbm, X, filename)
        feature_imp = DataFrame({'Value': gbm.feature_importance(importance_type='gain'), 'Feature': X.columns})
        return feature_imp

    def analyze(self, df: DataFrame):
        df = df.drop(['agent', 'bill', 'members', 'month', 'tariff', 'num_agents'], axis=1)

        behaviours = {'cost-driven': ['client', 'techno-solutionist'],
                      'consumption-driven': ['environmentalist', 'commited']}
        densities = {1: 'low density', 2: 'medium density', 3: 'high density'}
        income_types = {1:'low', 2:'medium', 3:'high'}
        results = dict()
        # Training for all behaviours
        results['all_behaviours'] = self.perform_gbm_analysis(df, 'all_behaviours')

        # Performing all relevant data combinations for analysis
        for key, behaviour_set in behaviours.items():
            filtered_by_behaviour = df.query("behaviour in @behaviour_set")
            logging.debug("Filtering by behaviour "+key)
            logging.debug(filtered_by_behaviour.shape)
            results[key] = self.perform_gbm_analysis(filtered_by_behaviour, key)
            for density, sdensity in densities.items():
                # Train all densities for each behaviour_set
                filename = '_'.join([key, sdensity]).replace(' ', '-')
                filtered_by_density = filtered_by_behaviour.query("density == @density")
                logging.debug("Filtering by density "+sdensity)
                logging.debug(filtered_by_density.shape)
                results[filename] = self.perform_gbm_analysis(filtered_by_density, filename)
                for income_type, sincome_type in income_types.items():
                    # Training for all income_types, combined by density
                    # Take into account that there is no data for certain density-income_type combinations
                    # (low-high_density, high-low_density, low-medium_density, high-medium_density)
                    # We will just do the query and, if dataframe size is 0, no training is done.
                    filename = '_'.join([key, sdensity, sincome_type]).replace(' ', '-')+'-income'
                    filtered_by_income_type = filtered_by_density.query("income_type == @income_type")
                    logging.debug("Filtering by income_type")
                    logging.debug(filtered_by_income_type.shape)
                    if not filtered_by_income_type.empty:
                        results[filename] = self.perform_gbm_analysis(filtered_by_income_type, filename)

        # Now we use results dict to generate scaled charts of feature importance
        max_value = max([df['Value'].max() for df in results.values()])
        for filename, feature_imp in results.items():
            self.plot_feature_importance(feature_imp, filename, max_value, 0.0)
    
    def consumption_charts_by_behaviour(self, df: DataFrame):
        behaviours = {'cost-driven': ['client', 'techno-solutionist'],
                      'consumption-driven': ['environmentalist', 'commited']}
        densities = {1:'low density', 2:'medium density', 3:'high density'}
        income_types = {1:'low', 2:'medium', 3:'high'}
        df_by_month = df[['month', 'consumption_per_capita (m3)', 'behaviour']]
        data = {}
        for key, behaviour_set in behaviours.items():
            filtered_by_behaviour = df_by_month.query("behaviour in @behaviour_set")
            aux = filtered_by_behaviour.drop(['behaviour'], axis=1)
            mean = aux.groupby(['month']).mean()
            std = aux.groupby(['month']).std()
            mean_1 = mean['consumption_per_capita (m3)'].to_numpy()
            std_1 = std['consumption_per_capita (m3)'].to_numpy()
            data[key] = {'mean': mean_1.copy(), 'std': std_1.copy()}
        self.plot_consumptions(data, 'by behaviour')
        
    def consumption_charts_by_behaviour_and_density(self, df: DataFrame):
        # By density and behaviour
        cost_driven = ['client', 'techno-solutionist']
        consumption_driven = ['commited', 'environmentalist']
        behaviours = {'cost-driven': cost_driven, 'consumption-driven': consumption_driven}
        densities = {'low density': 1, 'medium density': 2, 'high density': 3}
        df_by_month_density = df[['month', 'consumption_per_capita (m3)', 'density', 'behaviour']]
        data = {}
        for key in behaviours.keys():
            data[key] = {}


        for key, behaviour_set in behaviours.items():
            filtered_by_behaviour = df_by_month_density.query("behaviour in @behaviour_set")
            aux = filtered_by_behaviour.drop(['behaviour'], axis=1)
            for sdensity, density in densities.items():
                temp_df = aux[aux['density'] == density]
                filtered_by_density = temp_df.drop(['density'], axis=1)
                mean = filtered_by_density.groupby(['month']).mean()
                std = filtered_by_density.groupby(['month']).std()
                data[key][sdensity] = {'mean': mean['consumption_per_capita (m3)'].to_numpy(), 'std': std['consumption_per_capita (m3)'].to_numpy()}

        for key in behaviours.keys():
            self.plot_consumptions(data[key], title=' '+key)
    
    def consumption_charts_by_behaviour_density_and_income_type(self, df: DataFrame):
        # By behaviour, density and income_type
        cost_driven = ['client', 'techno-solutionist']
        consumption_driven = ['commited', 'environmentalist']
        behaviours = {'cost-driven': cost_driven, 'consumption-driven': consumption_driven}

        data = {}
        for key in behaviours.keys():
            data[key] = {}

        densities = {'low density': 1, 'medium density': 2, 'high density': 3}
        income_types = {'low': 1, 'medium': 2, 'high': 3}

        df_by_month_density_income_type = df[['month', 'consumption_per_capita (m3)', 'density', 'income_type', 'behaviour']]

        for key, behaviour_set in behaviours.items():
            aux = df_by_month_density_income_type.query("behaviour in @behaviour_set")
            filtered_by_behaviour = aux.drop(['behaviour'], axis=1)
            
            for sdensity, density in densities.items():
                filtered_by_behaviour_density = filtered_by_behaviour[filtered_by_behaviour['density'] == density]
                for sincome_type, income_type in income_types.items():
                    aux = filtered_by_behaviour_density[filtered_by_behaviour_density['income_type'] == income_type]
                    if not aux.empty:
                        final_filtered = aux.drop(['density', 'income_type'], axis=1)
                        mean = final_filtered.groupby(['month']).mean()
                        std = final_filtered.groupby(['month']).std()
                        data[key][sdensity+" "+sincome_type] = {'mean': mean['consumption_per_capita (m3)'].to_numpy(), 'std': std['consumption_per_capita (m3)'].to_numpy()}
                    else:
                        logging.debug('This combination is empty: '+sdensity+ " " + sincome_type + ' income')

        for key in behaviours.keys():
            self.plot_consumptions(data[key], title=' '+key)
