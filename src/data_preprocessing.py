import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler, OneHotEncoder



class PenguinDataPreprocessor:
    def __init__(self):
        pass
    
    def fit(self, X=None, y=None):
        return self
    
    def transform(self, X=None):
        if X is None:
            df = self.database_retrieval()
        else:
            df = X.copy()
            
        preprocessing_steps = [
            self.data_cleaning,
            self.data_scaling,
            self.data_encoding,
            
        ]
        
        for step in preprocessing_steps:
            df = step(df)
            
        return df
    
    def fit_transform(self, X=None, y=None):
        return self.fit(X, y).transform(X)



    def database_retrieval(self):
        
        df = pd.read_csv('data/penguins.csv')

        # The culmen is the beak lol

        return df



    def data_cleaning(self, df):

        # Converting the outlier values in 'flipper_length_mm' and "." in 'sex' to NaN and then imputing 

        df['flipper_length_mm'] = df['flipper_length_mm'].replace([5000.0, -132.0], np.nan)
        df['sex'] = df['sex'].replace(['.'], np.nan)

        numerical_columns = ['culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm', 'body_mass_g']

        for col in numerical_columns: 
            df[col] = df[col].fillna(df[col].median())
            
        df['sex'] = df['sex'].fillna(df['sex'].mode()[0])
        
        return df



    def data_scaling(self, df):

        # Scaling the rest of the columns and encoding 'sex'
        numerical_columns = ['culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm', 'body_mass_g']

        scaler = StandardScaler()
        df[numerical_columns] = scaler.fit_transform(df[numerical_columns])
        
        return df



    def data_encoding(self, df):
        ohe = OneHotEncoder(sparse_output=False)
        encoded_sex = ohe.fit_transform(df[['sex']])
        encoded_df = pd.DataFrame(encoded_sex, columns=ohe.get_feature_names_out(['sex']))
        df = pd.concat([df, encoded_df], axis=1)
        df = df.drop('sex', axis=1)
        
        return df