# Test the model with custom patient data
def test_custom_patient(patient_data):
    # Create DataFrame with same column order as training data
    test_patient = pd.DataFrame([patient_data])
    
    # Reorder columns to match training data (X columns)
    test_patient = test_patient.reindex(columns=X.columns, fill_value=0)
    
    # Apply same preprocessing as training data
    for col in categorical_cols:
        if col in test_patient.columns:
            try:
                test_patient[col] = label_encoders[col].transform([str(test_patient[col].iloc[0])])
            except ValueError:
                test_patient[col] = 0
    
    # Make prediction
    if best_model_name == 'Logistic Regression':
        test_scaled = scaler.transform(test_patient)
        probability = best_model.predict_proba(test_scaled)[0, 1]
    else:
        probability = best_model.predict_proba(test_patient)[0, 1]
    
    return probability

# Test example
example_patient = {
    'age': 45, 'bmi': 28.5, 'systolic_bp': 130, 'family_history_diabetes': 1,
    'gender': 'Male', 'smoking_status': 'Never', 'alcohol_consumption_per_week': 2,
    'physical_activity_minutes_per_week': 150, 'diet_score': 7.5, 'sleep_hours_per_day': 7.0,
    'screen_time_hours_per_day': 4.0, 'waist_to_hip_ratio': 0.85, 'diastolic_bp': 85,
    'heart_rate': 75, 'cholesterol_total': 200, 'hdl_cholesterol': 45, 'ldl_cholesterol': 120,
    'triglycerides': 150, 'ethnicity': 'White', 'education_level': 'College',
    'income_level': 'Middle', 'employment_status': 'Employed', 'hypertension_history': 0,
    'cardiovascular_history': 0
}

probability = test_custom_patient(example_patient)
print(f"Diabetes Risk Probability: {probability:.4f} ({probability*100:.2f}%)")
print(f"Risk Level: {'HIGH' if probability > 0.5 else 'LOW'}")