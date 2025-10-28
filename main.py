import pandas as pd
from pandas.core.common import random_state

#read csv file
df = pd.read_csv('stroke_risk_dataset.csv', encoding = 'latin-1')
df = df.drop(columns = ['At Risk (Binary)', 'Stroke Risk (%)'])

from sklearn.ensemble import RandomForestRegressor
import joblib
model = joblib.load('rfr_cv.joblib')

predictions = model.predict(df)

print(predictions)

results_df = pd.DataFrame({'Predicted': predictions})
results_df.to_csv('random_forest_predictions.csv', index=False)

df_with_predictions = df.copy()
df_with_predictions[' Predicted Stroke Risk (%)'] = results_df

print(df_with_predictions)

df_with_predictions.to_csv('dataset_with_predictions.csv', index=False)

