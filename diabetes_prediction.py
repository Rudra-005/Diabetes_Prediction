import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

def load_and_preprocess_data():
    """Load and preprocess the training and test data"""
    # Load data
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    
    # Separate features and target
    X = train_df.drop(['id', 'diagnosed_diabetes'], axis=1)
    y = train_df['diagnosed_diabetes']
    X_test = test_df.drop(['id'], axis=1)
    
    # Handle categorical variables
    categorical_cols = ['gender', 'ethnicity', 'education_level', 'income_level', 
                       'smoking_status', 'employment_status']
    
    # Combine train and test for consistent encoding
    combined_df = pd.concat([X, X_test], axis=0, ignore_index=True)
    
    # Label encode categorical variables
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        combined_df[col] = le.fit_transform(combined_df[col].astype(str))
        label_encoders[col] = le
    
    # Split back to train and test
    X_processed = combined_df.iloc[:len(X)]
    X_test_processed = combined_df.iloc[len(X):]
    
    return X_processed, y, X_test_processed, test_df['id']

def train_models(X, y):
    """Train multiple models and return the best one"""
    # Split data for validation
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    models = {
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10),
        'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=42, max_depth=6),
        'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000)
    }
    
    best_model = None
    best_score = 0
    best_scaler = None
    
    print("Training and evaluating models...")
    for name, model in models.items():
        if name == 'LogisticRegression':
            model.fit(X_train_scaled, y_train)
            y_pred_proba = model.predict_proba(X_val_scaled)[:, 1]
            current_scaler = scaler
        else:
            model.fit(X_train, y_train)
            y_pred_proba = model.predict_proba(X_val)[:, 1]
            current_scaler = None
        
        score = roc_auc_score(y_val, y_pred_proba)
        print(f"{name}: ROC-AUC = {score:.4f}")
        
        if score > best_score:
            best_score = score
            best_model = model
            best_scaler = current_scaler
    
    print(f"\nBest model: {type(best_model).__name__} with ROC-AUC = {best_score:.4f}")
    return best_model, best_scaler

def generate_predictions(model, scaler, X_test, test_ids):
    """Generate predictions for test data"""
    if scaler is not None:
        X_test_scaled = scaler.transform(X_test)
        predictions = model.predict_proba(X_test_scaled)[:, 1]
    else:
        predictions = model.predict_proba(X_test)[:, 1]
    
    # Create submission dataframe
    submission = pd.DataFrame({
        'id': test_ids,
        'diagnosed_diabetes': predictions
    })
    
    return submission

def main():
    """Main function to run the diabetes prediction pipeline"""
    print("Loading and preprocessing data...")
    X, y, X_test, test_ids = load_and_preprocess_data()
    
    print(f"Training data shape: {X.shape}")
    print(f"Test data shape: {X_test.shape}")
    print(f"Target distribution: {y.value_counts().to_dict()}")
    
    # Train models
    best_model, best_scaler = train_models(X, y)
    
    # Generate predictions
    print("\nGenerating predictions...")
    submission = generate_predictions(best_model, best_scaler, X_test, test_ids)
    
    # Save submission
    submission.to_csv('submission.csv', index=False)
    print(f"Submission saved to submission.csv")
    print(f"Prediction statistics:")
    print(f"Min: {submission['diagnosed_diabetes'].min():.4f}")
    print(f"Max: {submission['diagnosed_diabetes'].max():.4f}")
    print(f"Mean: {submission['diagnosed_diabetes'].mean():.4f}")

if __name__ == "__main__":
    main()