import pandas as pd
from pandas.core.common import random_state

#read csv file
df = pd.read_csv('stroke_risk_dataset.csv', encoding = 'latin-1')

"""
#Used to test read of csv file
print(df.head())
"""

#drop binary column
df = df.drop(columns = ['At Risk (Binary)'])

"""
#view df to test modifications
print(df.head())
"""

X = df.drop(['Stroke Risk (%)'], axis = 1)
y = df['Stroke Risk (%)']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 15)

from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor(random_state = 15)
rfr.fit(X_train, y_train)

"""
# To view the first few rows of the training features
print(X_test.head())

# To view the first few rows of the testing targets
print(y_test.head())
"""

#run rfr on test data
y_pred = rfr.predict(X_test)

#test accuracy of model using mean absolute avg, mean squared error, and r2 score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
mean_absolute_error(y_pred,y_test)
mean_squared_error(y_pred,y_test)
r2_score(y_pred, y_test)

print(mean_absolute_error(y_pred,y_test))
print(mean_squared_error(y_pred,y_test))
print(r2_score(y_pred,y_test))


#set parameters to train rfr
param_grid = {
    'n_estimators': [150, 175, 200],
    'max_depth': [10,20,30],
    'min_samples_split': [2,5,10],
    'min_samples_leaf': [1,5,10],
}

#create new rfr with parameters
from sklearn.model_selection import GridSearchCV
rfr_cv = GridSearchCV(estimator=rfr, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
rfr_cv.fit(X_train, y_train)

#run new model on test data
y_pred = rfr_cv.predict(X_test)

#test accuracy of new model using mean absolute avg, mean squared error, and r2 score
mean_absolute_error(y_pred,y_test)
mean_squared_error(y_pred,y_test)
r2_score(y_pred, y_test)

print(mean_absolute_error(y_pred,y_test))
print(mean_squared_error(y_pred,y_test))
print(r2_score(y_pred,y_test))

#save model
import joblib
joblib.dump(rfr_cv,'rfr_cv.joblib')

