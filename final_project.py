# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # House Price Prediction

# ## Importing Libraries

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np 
import pandas as pd 
import os
import warnings
warnings.filterwarnings('ignore')

# ## Data Analysis

train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')

train_df.info()

# ### Example of what an object datatype looks like

train_df.loc[:, 'LotShape']

# ### SalePrice vs GrLivArea scatter plot

data = pd.concat([train_df['SalePrice'], train_df['GrLivArea']], axis=1)
data.plot.scatter(x='GrLivArea', y='SalePrice', ylim=(0,900000));

# Positive correlation between SalePrice and livng area square footage

# ### SalePrice vs YearBuilt scatter plot

data = pd.concat([train_df['SalePrice'], train_df['YearBuilt']], axis=1)
data.plot.scatter(x='YearBuilt', y='SalePrice');

# Exponential like correlation between SalePrice and YearBuilt

# ### SalePrice vs OverallQual boxplot

data = pd.concat([train_df['SalePrice'], train_df['OverallQual']], axis=1)
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x='OverallQual', y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000);

# Positive correlation between SalePrice and OverallQual

# ### Correlation Heatmap between every feature

corrmat = train_df.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True);

# Correlation between features

# ### Correlation between top 10 most correlated features with respect to SalePrice

k = 10 #number of variables for heatmap
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(train_df[cols].values.T)
#sns.set(font_scale=1.25)

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()

# ## Data Processing

# Removing any features with missing data

total = train_df.isnull().sum().sort_values(ascending=False)
percent = (train_df.isnull().sum()/train_df.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(25)

train_df = train_df.drop((missing_data[missing_data['Total'] > 1]).index,1)
train_df = train_df.drop(train_df.loc[train_df['Electrical'].isnull()].index)
print("Number of missing data in dataframe:", train_df.isnull().sum().max())

# Removing outliers in GrLivArea

train_df = train_df[train_df['GrLivArea'] < 4000]

# Remove ID Feature

train_df = train_df.drop('Id', axis = 1)

# Convert categorical data to trainable parameters and remove duplicate data points

train = pd.get_dummies(train_df)
train = train.drop_duplicates()

# Preparing data for Model training

y = train['SalePrice']
x = train.drop('SalePrice', axis = 1)

# ## Model Exploration

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score, KFold

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

# ### Random Forest Regressor Model

# Hyperparameter tuning

n_est_params = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
max_depth_params = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
f_model_acc = [0, 0, 0]

'''for i in n_est_params:
    for j in max_depth_params:
        f_model = RandomForestRegressor(n_estimators=i, max_depth=j, bootstrap=True, oob_score=True)
        f_model.fit(X_train, y_train)
        print(f_model.oob_score_)
        if f_model.oob_score_ > f_model_acc[0]:
            f_model_acc[0] = f_model.oob_score_
            f_model_acc[1] = i
            f_model_acc[2] = j
'''

# +
#print("Highest acc:", f_model_acc[0], "with n_est:", f_model_acc[1], "and max_depth:", f_model_acc[2])
# -

# Final Random Forest Model

forest_model = RandomForestRegressor(n_estimators=70, max_depth=14)
forest_model.fit(X_train, y_train)

# ### XG Boost Regressor Model

# Hyperparameter tuning

n_est_params = [50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150]
max_depth_params = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
learning_rate = [0.01, 0.1, 0.2, 0.3]
x_model_acc = [0, 0, 0, 0]

'''for i in n_est_params:
    for j in max_depth_params:
        for k in learning_rate:
            x_model = XGBRegressor(n_estimators=i, max_depth=j, learning_rate=k)
            x_model.fit(X_train, y_train)
            kfold = KFold(n_splits=10, shuffle=True)
            kf_cv_scores = cross_val_score(x_model, X_train, y_train, cv=kfold)
            print(kf_cv_scores.mean())
            if kf_cv_scores.mean() > x_model_acc[0]:
                x_model_acc[0] = kf_cv_scores.mean()
                x_model_acc[1] = i
                x_model_acc[2] = j
                x_model_acc[3] = k
'''

# +
#print("Highest acc:", x_model_acc[0], "\nn_est:", x_model_acc[1], "\nmax_depth:", x_model_acc[2], "\nlearning rate:", x_model_acc[3])
# -

# Final XG Boost Model

xg_model = XGBRegressor(n_estimators=120, max_depth=5, learning_rate=0.1)
xg_model.fit(X_train, y_train)

# ## Deep Neural Network Model

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam
from keras.callbacks import EarlyStopping 


# +
model = Sequential()

model = Sequential()
model.add(Dense(100, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(1))
# Compile model
model.compile(optimizer=Adam(learning_rate=0.001), loss = 'mse')
# -

early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
history = model.fit(x=X_train,y=y_train,
          validation_split=0.1,
          batch_size=128,epochs=400)


losses = pd.DataFrame(model.history.history)
losses.plot()

model.summary()

# ## Results

# ### Predicted Values vs True Values for all models

# +
f, ax = plt.subplots(2, 3, figsize=(20,15))
ax[0,0].plot(y_train, y_train, 'r-')
ax[0,0].set(title='Random Forest Model Training Data Accuracy', xlabel='True Values', ylabel='Predicted Values')
ax[0,0].scatter(y_train, forest_model.predict(X_train))

ax[1,0].plot(y_test, y_test, 'r-')
ax[1,0].set(title='Random Forest Model Test Data Accuracy', xlabel='True Values', ylabel='Predicted Values')
ax[1,0].scatter(y_test, forest_model.predict(X_test))

ax[0,1].plot(y_train, y_train, 'r-')
ax[0,1].set(title='XG Boost Model Training Data Accuracy', xlabel='True Values', ylabel='Predicted Values')
ax[0,1].scatter(y_train, xg_model.predict(X_train))

ax[1,1].plot(y_test, y_test, 'r-')
ax[1,1].set(title='XG Boost Model Test Data Accuracy', xlabel='True Values', ylabel='Predicted Values')
ax[1,1].scatter(y_test, xg_model.predict(X_test))

ax[0,2].plot(y_train, y_train, 'r-')
ax[0,2].set(title='DNN Model Training Data Accuracy', xlabel='True Values', ylabel='Predicted Values')
ax[0,2].scatter(y_train, model.predict(X_train))

ax[1,2].plot(y_test, y_test, 'r-')
ax[1,2].set(title='DNN Model Test Data Accuracy', xlabel='True Values', ylabel='Predicted Values')
ax[1,2].scatter(y_test, model.predict(X_test))
# -

# ### Prediciton metrics for all models

# +
y_pred_forest = forest_model.predict(X_test)
y_pred_xg = xg_model.predict(X_test)
y_pred_dnn = model.predict(X_test)
from sklearn import metrics
print('Random Forest Model Metrics')
print('MAE:', metrics.mean_absolute_error(y_test, y_pred_forest))  
print('MSE:', metrics.mean_squared_error(y_test, y_pred_forest))  
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred_forest)))
print('VarScore:',metrics.explained_variance_score(y_test,y_pred_forest), '\n')

print('XG Boost Model Metrics')
print('MAE:', metrics.mean_absolute_error(y_test, y_pred_xg))  
print('MSE:', metrics.mean_squared_error(y_test, y_pred_xg))  
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred_xg)))
print('VarScore:',metrics.explained_variance_score(y_test,y_pred_xg), '\n')

print('Deep Neural Network Model Metrics')
print('MAE:', metrics.mean_absolute_error(y_test, y_pred_dnn))  
print('MSE:', metrics.mean_squared_error(y_test, y_pred_dnn))  
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred_dnn)))
print('VarScore:',metrics.explained_variance_score(y_test,y_pred_dnn))
# -


