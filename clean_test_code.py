# Test the model with custom patient data
def test_custom_patient(patient_data):
    """Test model with custom patient data"""
    # Create DataFrame from input
    test_patient = pd.DataFrame([patient_data])
    
    # Apply same preprocessing as training data
    for col in categorical_cols:
        if col in test_patient.columns:
            # Use the same label encoders from training
            try:
                test_patient[col] = label_encoders[col].transform([str(test_patient[col].iloc[0])])
            except ValueError:
                print(f"Warning: Unknown category '{test_patient[col].iloc[0]}' for {col}. Using most common category.")
                test_patient[col] = 0  # Default to first encoded value
    
    # Make prediction
    if best_model_name == 'Logistic Regression':
        test_scaled = scaler.transform(test_patient)
        probability = best_model.predict_proba(test_scaled)[0, 1]
        prediction = best_model.predict(test_scaled)[0]
    else:
        probability = best_model.predict_proba(test_patient)[0, 1]
        prediction = best_model.predict(test_patient)[0]
    
    return prediction, probability

# Example patient data (modify these values to test different scenarios)
example_patient = {
    'age': 45,
    'alcohol_consumption_per_week': 2,
    'physical_activity_minutes_per_week': 150,
    'diet_score': 7.5,
    'sleep_hours_per_day': 7.0,
    'screen_time_hours_per_day': 4.0,
    'bmi': 28.5,
    'waist_to_hip_ratio': 0.85,
    'systolic_bp': 130,
    'diastolic_bp': 85,
    'heart_rate': 75,
    'cholesterol_total': 200,
    'hdl_cholesterol': 45,
    'ldl_cholesterol': 120,
    'triglycerides': 150,
    'gender': 'Male',
    'ethnicity': 'White',
    'education_level': 'College',
    'income_level': 'Middle',
    'smoking_status': 'Never',
    'employment_status': 'Employed',
    'family_history_diabetes': 1,
    'hypertension_history': 0,
    'cardiovascular_history': 0
}

# Test the example patient
prediction, probability = test_custom_patient(example_patient)

print("Custom Patient Test Results:")
print("=" * 40)
print(f"Patient Profile:")
for key, value in example_patient.items():
    print(f"  {key}: {value}")

print(f"\nPrediction Results:")
print(f"  Diabetes Risk: {'HIGH' if prediction == 1 else 'LOW'}")
print(f"  Probability: {probability:.4f} ({probability*100:.2f}%)")

if probability > 0.7:
    risk_level = "HIGH RISK"
elif probability > 0.4:
    risk_level = "MODERATE RISK"
else:
    risk_level = "LOW RISK"

print(f"  Risk Level: {risk_level}")