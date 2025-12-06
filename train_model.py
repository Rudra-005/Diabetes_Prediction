import pandas as pd
import pickle
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder

print("Loading data...")
train_df = pd.read_csv('train.csv')

print("Preprocessing data...")
X = train_df.drop(['id', 'diagnosed_diabetes'], axis=1)
y = train_df['diagnosed_diabetes']

categorical_cols = ['gender', 'ethnicity', 'education_level', 'income_level', 
                   'smoking_status', 'employment_status']

label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le

print("Training model...")
model = GradientBoostingClassifier(n_estimators=100, random_state=42, max_depth=6)
model.fit(X, y)

print("Saving model...")
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('encoders.pkl', 'wb') as f:
    pickle.dump(label_encoders, f)

with open('feature_columns.pkl', 'wb') as f:
    pickle.dump(X.columns.tolist(), f)

print("✓ Model trained and saved successfully!")
print(f"Features: {len(X.columns)}")
print(f"Training samples: {len(X)}")