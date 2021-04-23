import pandas as pd
import numpy as np

class EZPrep:
    def __init__(self, data):
        self.raw_data = data
        self.prep_data = data

    @property
    def hello_world(self):
        print("You're now with EZPREP!!")
    
    def check_up(self):
        return self.raw_data.isna().sum()

    def drop_col(self, columns):
        self.prep_data = self.raw_data.drop(columns, axis=1).copy()

    def dropna(self):
        self.prep_data = self.raw_data.dropna().copy()
        return self.prep_data

    def to_datetime(self, column):
        self.prep_data[column] = pd.to_datetime(self.prep_data[column])
        return self.prep_data

    def to_frame(self):
        return self.prep_data