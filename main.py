import pandas as pd

from pandas.core.common import random_state

#read csv file
df = pd.read_csv('stroke_risk_dataset.csv', encoding = 'latin-1')
df = df.drop(columns = ['At Risk (Binary)', 'Stroke Risk (%)'])

from sklearn.ensemble import RandomForestRegressor
import joblib
model = joblib.load('rfr_cv.joblib')

predictions = model.predict(df)

results_df = pd.DataFrame({'Predicted': predictions})
results_df.to_csv('random_forest_predictions.csv', index=False)

df_with_predictions = df.copy()
df_with_predictions['Predicted Stroke Risk (%)'] = results_df

df_with_predictions.to_csv('dataset_with_predictions.csv', index=False)

print('Stroke Risk Predictions:')
print(df_with_predictions)
"""
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

tree_to_plot = model.best_estimator_.estimators_[0]

plt.figure(figsize=(20, 10)) # Adjust figure size for better readability
plot_tree(tree_to_plot,
          filled=True,
          rounded=True)
plt.title("Individual Decision Tree from Random Forest Regressor")
plt.show()
"""
#Create data frame from csv
df1 = pd.read_csv('dataset_with_predictions.csv')

#Categorize High risk patients as patients with risk level higher than 75
High_Risk = df1[df1['Predicted Stroke Risk (%)'] > 75]
HR_count = len(High_Risk.index)

#Categorize Medium risk patients as those with risk between 50 and 75
Med_Risk = df1[(df1['Predicted Stroke Risk (%)'] > 50) & (df1['Predicted Stroke Risk (%)'] < 75)]
MR_count = len(Med_Risk.index)

#Categorize low risk patients as those with risk less than 50
Low_Risk = df1[df1['Predicted Stroke Risk (%)'] <50]
LR_count = len(Low_Risk.index)

#Create Bar graph to show number of patients at each risk level
import matplotlib.pyplot as plt

Risk = ['Low_Risk', 'Med_Risk', 'High_Risk']
Patients = [24821, 39858, 5316]

plt.bar(Risk, Patients)

df2 = df1.head(500)

plt.xlabel('Risk Level')
plt.ylabel('Number of Patients')
plt.title('Stroke Risk')
plt.show()

plt.scatter(df2['Predicted Stroke Risk (%)'], df2['Age'])

plt.xlabel('Stroke Risk')
plt.ylabel('Age')
plt.title(f'Scatter Plot of Stroke Risk vs Age')

# Display the plot
plt.show()

"""
#Create Bar graph to show number of patients with high or low blood pressure
High_BP = df1[df1['High Blood Pressure'] == 1]
HBP_Count = len(High_BP.index)
print(f"Number of Patients with High BP: {HBP_Count}")

Normal_BP = df1[df1['High Blood Pressure'] == 0]
NBP_Count = len(Normal_BP.index)
print(f"Number of Patients with Normal BP: {NBP_Count}")

Blood_Pressure = ['High_BP', 'Normal_BP']
BP_Patients = [35045, 34955]

plt.bar(Blood_Pressure, BP_Patients)

plt.xlabel('Blood Pressure')
plt.ylabel('Number of Patients')
plt.title('High BP vs. Normal BP')
plt.show()
"""

#Create 10 patients for scatter plot
pt1 = df1.iloc[1] #create data frame of a single patient
pt1_count = (pt1 == 1).sum() #Find total number of Symptoms the patient has
pt1_SR = pt1['Predicted Stroke Risk (%)'] #find patients stroke risk prediction

pt2 = df1.iloc[2]
pt2_count = (pt2 == 1).sum()
pt2_SR = pt2['Predicted Stroke Risk (%)']

pt3 = df1.iloc[3]
pt3_count = (pt3 == 1).sum()
pt3_SR = pt3['Predicted Stroke Risk (%)']

pt4 = df1.iloc[4]
pt4_count = (pt4 == 1).sum()
pt4_SR = pt4['Predicted Stroke Risk (%)']

pt5 = df1.iloc[5]
pt5_count = (pt5 == 1).sum()
pt5_SR = pt5['Predicted Stroke Risk (%)']

pt6 = df1.iloc[6]
pt6_count = (pt6 == 1).sum()
pt6_SR = pt6['Predicted Stroke Risk (%)']

pt7 = df1.iloc[7]
pt7_count = (pt7 == 1).sum()
pt7_SR = pt7['Predicted Stroke Risk (%)']

pt8 = df1.iloc[8]
pt8_count = (pt8 == 1).sum()
pt8_SR = pt8['Predicted Stroke Risk (%)']

pt9 = df1.iloc[9]
pt9_count = (pt9 == 1).sum()
pt9_SR = pt9['Predicted Stroke Risk (%)']

pt10 = df1.iloc[10]
pt10_count = (pt10 == 1).sum()
pt10_SR = pt10['Predicted Stroke Risk (%)']

#Create Scatter plot Stroke Risk vs. Number of symptoms
x_vals = [pt1_SR, pt2_SR, pt3_SR, pt4_SR, pt5_SR, pt6_SR, pt7_SR, pt8_SR, pt9_SR, pt10_SR]
y_vals = [pt1_count, pt2_count, pt3_count, pt4_count, pt5_count, pt6_count,pt7_count,pt8_count,pt9_count,pt10_count]
plt.scatter(x_vals, y_vals)

plt.xlabel('Stroke Risk')
plt.ylabel('Number of Symptoms')
plt.title(f'Scatter Plot of Stroke Risk vs Symptoms')

# Display the plot
plt.show()

"""
n = 107
print('\nPatient ', n , ':')
print(df1.iloc[n])
"""

#Prints list of High Risk Patients sorted from high to low based on risk level
High_Risk_Sorted = High_Risk.sort_values(by='Predicted Stroke Risk (%)', ascending=False)
print('\nHigh Risk Patients')
print('Number of High Risk Patients:', HR_count)

print(High_Risk_Sorted.head(25))

#Prompts user for input, prints patient information based on input
pt_num = input("Enter Patient Number : ")
pt_num_input = int(pt_num)
print('\nPatient ', pt_num_input , ':')
print(df1.iloc[pt_num_input])