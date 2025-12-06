import pandas as pd
import pickle
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

print("Loading data...")
train_df = pd.read_csv('train.csv')

# Separate features and target
X = train_df.drop(['id', 'diagnosed_diabetes'], axis=1)
y = train_df['diagnosed_diabetes']

# Store feature columns
feature_columns = X.columns.tolist()

# Encode categorical variables
categorical_cols = ['gender', 'ethnicity', 'education_level', 'income_level', 
                   'smoking_status', 'employment_status']
label_encoders = {}

for col in categorical_cols:
    if col in X.columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le

print("Training model...")
model = GradientBoostingClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

print("Saving model and encoders...")
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('encoders.pkl', 'wb') as f:
    pickle.dump(label_encoders, f)

with open('feature_columns.pkl', 'wb') as f:
    pickle.dump(feature_columns, f)

print("✅ Model saved successfully!")
print("Files created: model.pkl, encoders.pkl, feature_columns.pkl")
