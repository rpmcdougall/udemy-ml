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

# Handle missing data
# imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
# imputer = imputer.fit(feature_matrix[:, 1:3])
# feature_matrix[:, 1:3] = imputer.transform(feature_matrix[:, 1:3])

"""
Encode Categorical Variables using One Hot Encoding
"""
# # Country
# label_encoder_feat = LabelEncoder()
# feature_matrix[:, 0] = label_encoder_feat.fit_transform(feature_matrix[:, 0])
# ohe = OneHotEncoder(categorical_features=[0])
# feature_matrix = ohe.fit_transform(feature_matrix).toarray()
#
# # Purchased
# label_encoder_dep = LabelEncoder()
# dep_var_vec = label_encoder_dep.fit_transform(dep_var_vec)

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




