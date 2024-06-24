import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import sqlite3
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

# Load the data
db_file = "hearts.db"
table_name = "hearts"

try:
    conn = sqlite3.connect(db_file)
    df = pd.read_sql(f"SELECT * FROM {table_name}", conn)
except sqlite3.Error as e:
    print(f"Error connecting to database: {e}")
finally:
    conn.close()

# Handle missing values
df.fillna(df.mean(), inplace=True)

# Encode categorical variables
categorical_columns = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
for column in categorical_columns:
    df[column] = LabelEncoder().fit_transform(df[column])

# Define features and target variable
X = df.drop(columns=['target'])
y = df['target']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Models
log_reg = LogisticRegression()
rand_forest = RandomForestClassifier()
svm = SVC(probability=True)

# Train models
log_reg.fit(X_train, y_train)
rand_forest.fit(X_train, y_train)
svm.fit(X_train, y_train)

# Perform predictions
y_pred_log_reg = log_reg.predict(X_test)
y_pred_rand_forest = rand_forest.predict(X_test)
y_pred_svm = svm.predict(X_test)

# Evaluate models
print("Logistic Regression:")
print("Accuracy:", accuracy_score(y_test, y_pred_log_reg))
print(confusion_matrix(y_test, y_pred_log_reg))
print(classification_report(y_test, y_pred_log_reg))

print("\nRandom Forest:")
print("Accuracy:", accuracy_score(y_test, y_pred_rand_forest))
print(confusion_matrix(y_test, y_pred_rand_forest))
print(classification_report(y_test, y_pred_rand_forest))

print("\nSupport Vector Machine:")
print("Accuracy:", accuracy_score(y_test, y_pred_svm))
print(confusion_matrix(y_test, y_pred_svm))
print(classification_report(y_test, y_pred_svm))

# Save the best model
best_model = rand_forest
joblib.dump(best_model, 'best_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
