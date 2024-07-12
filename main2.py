import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import pickle

# Load the dataset
file_path = './dataset/data_2.csv'
data = pd.read_csv(file_path)

# Remove leading spaces from column names
data.columns = data.columns.str.strip()

# Encode the target variable
le = LabelEncoder()
data['OUTPUT'] = le.fit_transform(data['OUTPUT'])

# Split the dataset
X = data[['HR', 'RESP', 'SpO2', 'TEMP']]
y = data['OUTPUT']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the SVC model
svc = SVC()
svc.fit(X_train, y_train)

# Predict and evaluate
y_pred = svc.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print(f'Classification Report:\n{report}')

# Save the trained model
with open('./models/SVC_model_Pred.pkl', 'wb') as model_file:
    pickle.dump(svc, model_file)

# Guardar el scaler
with open('./models/SVC_scaler_Pred.pkl', 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)
