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

# ## Importing Libraries

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np 
import pandas as pd 
import os

# ## Data Analysis

train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')

train_df.info()

data = pd.concat([train_df['SalePrice'], train_df['GrLivArea']], axis=1)
data.plot.scatter(x='GrLivArea', y='SalePrice', ylim=(0,900000));

# Positive correlation between SalePrice and livng area square footage

data = pd.concat([train_df['SalePrice'], train_df['YearBuilt']], axis=1)
data.plot.scatter(x='YearBuilt', y='SalePrice');

# Exponential like correlation between SalePrice and YearBuilt

data = pd.concat([train_df['SalePrice'], train_df['OverallQual']], axis=1)
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x='OverallQual', y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000);

# Positive correlation between SalePrice and OverallQual

corrmat = train_df.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True);

# Corrleation between features

k = 10 #number of variables for heatmap
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(train_df[cols].values.T)
#sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()

# Correlation between top 10 most correlated features with respect to SalePrice

sns.set()
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(train_df[cols], height = 2.5)
plt.show();

# ## Data Processing

total = train_df.isnull().sum().sort_values(ascending=False)
percent = (train_df.isnull().sum()/train_df.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(25)

train_df = train_df.drop((missing_data[missing_data['Total'] > 1]).index,1)
train_df = train_df.drop(train_df.loc[train_df['Electrical'].isnull()].index)
print("Number of missing data in dataframe:", train_df.isnull().sum().max())

# Removing features with missing data

train = pd.get_dummies(train_df)

# Converting categorical data to numerical data

# ## Trying Base Models

from sklearn.linear_model import LinearRegression

# +
y = train['SalePrice']

x = train.drop('SalePrice', axis = 1)
# -

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

lr = LinearRegression()

lr.fit(X_train, y_train)

print("Training set score: {:.2f}".format(lr.score(X_train, y_train))) 
print("Test set score: {:.2f}".format(lr.score(X_test, y_test)))


from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

decision_model = DecisionTreeRegressor()  
decision_model.fit(X_train, y_train)

print("Training set score: {:.2f}".format(decision_model.score(X_train, y_train))) 
print("Test set score: {:.2f}".format(decision_model.score(X_test, y_test)))

forest_model = RandomForestRegressor(n_estimators=100, max_depth=10)
forest_model.fit(X_train, y_train)

print("Training set score: {:.2f}".format(forest_model.score(X_train, y_train))) 
print("Test set score: {:.2f}".format(forest_model.score(X_test, y_test)))

xg_model = XGBRegressor(n_estimators=100)
xg_model.fit(X_train, y_train)

print("Training set score: {:.2f}".format(xg_model.score(X_train, y_train))) 
print("Test set score: {:.2f}".format(xg_model.score(X_test, y_test)))

p1 = max(max(forest_model.predict(X_train)), max(y_train))
p2 = min(min(forest_model.predict(X_train)), min(y_train))
plt.plot([p1, p2], [p1, p2], 'r-')
plt.title('Training Data Accuracy')
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.scatter(y_train, forest_model.predict(X_train))

p1 = max(max(forest_model.predict(X_test)), max(y_test))
p2 = min(min(forest_model.predict(X_test)), min(y_test))
plt.plot([p1, p2], [p1, p2], 'r-')
plt.title('Test Data Accuracy')
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.scatter(y_test, forest_model.predict(X_test))

# # DNN

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

loss_df = pd.DataFrame(model.history.history)
loss_df.plot(figsize=(12,8))

y_pred = model.predict(X_test)
from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(y_test, y_pred))  
print('MSE:', metrics.mean_squared_error(y_test, y_pred))  
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('VarScore:',metrics.explained_variance_score(y_test,y_pred))
# Visualizing Our predictions
fig = plt.figure(figsize=(10,5))
plt.scatter(y_test,y_pred)
# Perfect predictions
plt.plot(y_test,y_test,'r')


