import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# Import Data
dataset = pd.read_csv('data/p2_linear_regression/Salary_Data.csv')

# Load data set
feature_matrix = dataset.iloc[:, :-1].values
dep_var_vec = dataset.iloc[:, 1].values

"""
Split data into training and test set
"""
feat_train, feat_test, dep_train, dep_test = train_test_split(feature_matrix, dep_var_vec, test_size=1/3, random_state=0)

"""
Scale Features, Not needed for simple linear regression
"""
# sc_feat = StandardScaler()
# feat_train = sc_feat.fit_transform(feat_train)
# feat_test = sc_feat.transform(feat_test)

'''
Fitting Simple Linear Regression to Training Set
'''
regressor = LinearRegression()
regressor.fit(feat_train, dep_train)


'''›
Predicting the test set results
'''
dep_pred = regressor.predict(feat_test)


'''
Visualize Results of Training Set
'''

plt.scatter(feat_train, dep_train, color='red')
plt.plot(feat_train, regressor.predict(feat_train), color='blue')
plt.title('Salary vs Experience (Training Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

'''
Visualize Results of Test Set
'''
plt.scatter(feat_test, dep_test, color='red')
plt.plot(feat_train, regressor.predict(feat_train), color='blue')
plt.title('Salary vs Experience (Test Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
