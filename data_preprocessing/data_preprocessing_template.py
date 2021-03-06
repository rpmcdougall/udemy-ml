import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split


# Import Data
dataset = pd.read_csv('data/Data.csv')

# Load data set
feature_matrix = dataset.iloc[:, :-1].values
dep_var_vec = dataset.iloc[:, 3].values

"""
Split data into training and test set
"""
feat_train, feat_test, dep_train, dep_test = train_test_split(feature_matrix, dep_var_vec, test_size=0.2, random_state=0)

"""
Scale Features
"""
# sc_feat = StandardScaler()
# feat_train = sc_feat.fit_transform(feat_train)
# feat_test = sc_feat.transform(feat_test)




