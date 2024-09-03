import os
import time
import tempfile
import pandas as pd
from preprocessor import Preprocessor
FILE_BATCH_SIZE: int = 100


"""This class takes care of merging all the CSV files resulting from Netlogo experiment."""
class FileMerger(object):

    def __init__(self, preprocessor: Preprocessor):
        self.preprocessor = preprocessor

    def __process_one_csv(self, filename: str):
        print('__process_one_csv: ' + filename)
        df_in = pd.read_csv(filename)
        #df = self.preprocessor.add_columns(filename, df_in)
        return df_in

    def __generate_intermediate_csvs(self, path_in, path_out):
        df_csv_append = pd.DataFrame()
        files = os.listdir(path_in)
        csvs = filter(lambda f: f.endswith('.csv'), files)
        i = 1
        for filename in csvs:
            full_filename = os.path.join(path_in, filename)
            file_df = self.__process_one_csv(full_filename)
            df_csv_append = pd.concat([df_csv_append, file_df], ignore_index=True)
            if i % FILE_BATCH_SIZE == 0:
                out_filename = os.path.join(path_out, str(i) + '.csv')
                df_csv_append.to_csv(out_filename, index=False)
                df_csv_append = pd.DataFrame()
            i += 1
        if not df_csv_append.empty:
            out_filename = os.path.join(path_out, str(i) + '.csv')
            df_csv_append.to_csv(out_filename, index=False)

    def merge_csvs(self, path_in: str, path_final: str):
        with tempfile.TemporaryDirectory() as dirname:
            print(dirname)
            self.__generate_intermediate_csvs(path_in, dirname)
            df_csv_append = pd.DataFrame()
            files = os.listdir(dirname)
            csvs = filter(lambda f: f.endswith('.csv'), files)
            start_time = time.time()
            for file in csvs:
                df = pd.read_csv(os.path.join(dirname, file))
                df_csv_append = pd.concat([df_csv_append, df], ignore_index=True)
                print(file, " Time: ", time.time() - start_time)
                start_time = time.time()
            df_csv_append.to_csv(os.path.join(path_final, 'all.csv'), index=False)
