import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
from imblearn.combine import SMOTEENN
from sklearn.feature_selection import SelectKBest, f_classif
import pickle

df = pd.read_csv("C:/Users/srika/Downloads/diabetes.csv")

df.head()
df.tail()
df.info()
df.describe()

plt.figure(figsize=(12, 6))
df.iloc[:, :-1].boxplot()
plt.xticks(rotation=90)
plt.title("Feature Distribution Before Outlier Removal")
plt.show()
plt.figure(figsize=(6,4))
sns.countplot(x=df['Outcome'], palette='coolwarm')
plt.title('Class Distribution')
plt.show()

for feature in df.columns[:-1]:
    Q1 = df[feature].quantile(0.25)
    Q3 = df[feature].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df = df[(df[feature] >= lower_bound) & (df[feature] <= upper_bound)]
df.describe()
correlation = df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Feature Correlation Heatmap")
plt.show()

X = df.drop(columns=['Outcome'])
y = df['Outcome']
k_best = SelectKBest(score_func=f_classif, k=3)
X_selected = k_best.fit_transform(X, y)
selected_features = X.columns[k_best.get_support()].tolist()
print("Top 3 Features using SelectKBest:", selected_features)
X = df[selected_features]
y = df['Outcome']
smote_enn = SMOTEENN()
X_resampled, y_resampled = smote_enn.fit_resample(X, y)
plt.figure(figsize=(6,4))
sns.countplot(x=y_resampled, palette='coolwarm')
plt.title('Balanced Class Distribution After SMOTE-ENN')
plt.show()

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("\nModel Accuracy:", accuracy)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['No Diabetes', 'Diabetes'], yticklabels=['No Diabetes', 'Diabetes'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Saving the model using pickle
with open('diabetes_rf_model.pkl', 'wb') as f:
    pickle.dump((scaler, model), f)

print("\nModel saved as 'diabetes_rf_model.pkl'")

from flask import Flask, request, jsonify
import pickle
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the trained model and scaler
with open('diabetes_rf_model.pkl', 'rb') as f:
    scaler, model = pickle.load(f)

@app.route('/')
def home():
    return "Diabetes Prediction API is Running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data
        data = request.get_json()
        
        # Convert data to numpy array and reshape
        features = np.array(data['features']).reshape(1, -1)
        
        # Scale input
        features_scaled = scaler.transform(features)
        
        # Make prediction
        prediction = model.predict(features_scaled)[0]
        
        # Return response
        return jsonify({'prediction': int(prediction), 'message': 'Diabetes' if prediction == 1 else 'No Diabetes'})
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)

from flask import Flask
app = Flask(__name__)

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)