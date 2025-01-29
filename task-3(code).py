# Libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm

# Input Dataset
data = pd.DataFrame({
    'Outlook': ['sunny', 'sunny', 'overcast', 'rain', 'rain', 'rain', 'overcast', 'sunny', 'sunny', 'rain', 'sunny', 'overcast', 'overcast', 'rain'],
    'Temperature': ['hot', 'hot', 'hot', 'mild', 'cool', 'cool', 'cool', 'mild', 'cool', 'mild', 'mild', 'mild', 'hot', 'mild'],
    'Humidity': ['high', 'high', 'high', 'high', 'normal', 'normal', 'normal', 'high', 'normal', 'normal', 'normal', 'high', 'normal', 'high'],
    'Windy': [False, True, False, False, False, True, True, False, False, False, True, True, False, True],
    'Class': ['-', '-', '+', '+', '+', '-', '+', '-', '+', '+', '+', '+', '+', '-']
})

# Encode Categorical Data
label_encoders = {}
for col in ['Outlook', 'Temperature', 'Humidity', 'Class']:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# One-Hot Encoding for 'Outlook'
ohe = OneHotEncoder()
outlook_encoded = ohe.fit_transform(data[['Outlook']]).toarray()
outlook_df = pd.DataFrame(outlook_encoded, columns=ohe.get_feature_names_out(['Outlook']))

# Combine Encoded Data
s = pd.concat([outlook_df, data[['Temperature', 'Humidity', 'Windy']], data['Class']], axis=1)

# Train-Test Split
X = s.iloc[:, :-1]  # Features
y = s.iloc[:, -1:]  # Target
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

# Train Linear Regression Model
from sklearn.linear_model import LogisticRegression

# Train Logistic Regression Model
logistic_model = LogisticRegression()
logistic_model.fit(x_train, y_train.values.ravel())  # values.ravel() flattens the y_train array.

# Predictions with Probabilities
y_pred_proba = logistic_model.predict_proba(x_test)

# For a specific test case <sunny, cool, high, strong>
test_case = pd.DataFrame({
    'Outlook': ['sunny'],
    'Temperature': ['cool'],
    'Humidity': ['high'],
    'Windy': [True]
})

# Encode Test Case
for col in ['Outlook', 'Temperature', 'Humidity']:
    test_case[col] = label_encoders[col].transform(test_case[col])

test_case_encoded = pd.DataFrame(ohe.transform(test_case[['Outlook']]).toarray(), columns=ohe.get_feature_names_out(['Outlook']))
test_case_encoded = pd.concat([test_case_encoded, test_case[['Temperature', 'Humidity', 'Windy']]], axis=1)

# Predict Probability
test_case_proba = logistic_model.predict_proba(test_case_encoded)

# Output
print("\nPredicted Probabilities for Test Cases (y_test):")
print(y_pred_proba)

print("\nPrediction for <sunny, cool, high, strong>:")
print(f"Probability of Playing Tennis: {test_case_proba[0][1]:.2f}")
