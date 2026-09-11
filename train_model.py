import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

# Load the dataset
df = pd.read_csv('heart.csv')

X = df.drop(columns=['HeartDisease'])
y = df['HeartDisease']

# 1. Apply one-hot encoding for categorical string columns
X_encoded = pd.get_dummies(X, drop_first=True)

# 2. Add Scaling
scaler = StandardScaler()
num_cols = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
X_encoded[num_cols] = scaler.fit_transform(X_encoded[num_cols])

# 3. Altered split ratio and random state
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.25, random_state=73)

# 4. Swap Random Forest for Gradient Boosting
model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=73)
model.fit(X_train, y_train)

# Save artifacts
joblib.dump(model, 'gb_classifier.pkl')
joblib.dump(list(X_encoded.columns), 'feature_names.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("Model trained and files saved successfully!")