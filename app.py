import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
import time

# Page config
st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon="🏥",
    layout="wide"
)

# Custom CSS for animations
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #FF4B4B;
        text-align: center;
        animation: fadeIn 1s;
    }
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    .stProgress > div > div > div > div {
        background-color: #FF4B4B;
    }
    .risk-high {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a6f 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        font-size: 1.5rem;
        animation: pulse 2s infinite;
    }
    .risk-low {
        background: linear-gradient(135deg, #51cf66 0%, #37b24d 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        font-size: 1.5rem;
        animation: pulse 2s infinite;
    }
    .risk-moderate {
        background: linear-gradient(135deg, #ffd43b 0%, #fab005 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        font-size: 1.5rem;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.05); }
    }
</style>
""", unsafe_allow_html=True)

# Load model and encoders
@st.cache_resource
def load_model():
    try:
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('encoders.pkl', 'rb') as f:
            label_encoders = pickle.load(f)
        with open('feature_columns.pkl', 'rb') as f:
            feature_columns = pickle.load(f)
        return model, label_encoders, feature_columns
    except:
        return None, None, None

model, label_encoders, feature_columns = load_model()

# Header
st.markdown('<h1 class="main-header">🏥 Diabetes Risk Predictor</h1>', unsafe_allow_html=True)
st.markdown("---")

if model is None:
    st.error("⚠️ Model not loaded. Please ensure train.csv is in the directory.")
else:
    # Sidebar
    st.sidebar.header("📋 Patient Information")
    st.sidebar.markdown("Fill in the details below:")
    
    # Input fields
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("👤 Demographics")
        age = st.slider("Age", 18, 100, 45)
        gender = st.selectbox("Gender", ["Male", "Female"])
        ethnicity = st.selectbox("Ethnicity", ["White", "Hispanic", "Black", "Asian", "Other"])
        education = st.selectbox("Education", ["Highschool", "College", "Graduate", "Other"])
        income = st.selectbox("Income Level", ["Lower", "Lower-Middle", "Middle", "Upper-Middle", "Upper"])
        employment = st.selectbox("Employment", ["Employed", "Unemployed", "Retired", "Self-Employed"])
    
    with col2:
        st.subheader("🏃 Lifestyle")
        alcohol = st.slider("Alcohol (drinks/week)", 0, 20, 2)
        physical_activity = st.slider("Physical Activity (min/week)", 0, 500, 150)
        diet_score = st.slider("Diet Score (0-10)", 0.0, 10.0, 7.5, 0.1)
        sleep = st.slider("Sleep (hours/day)", 4.0, 12.0, 7.0, 0.1)
        screen_time = st.slider("Screen Time (hours/day)", 0.0, 16.0, 4.0, 0.1)
        smoking = st.selectbox("Smoking Status", ["Never", "Former", "Current"])
    
    with col3:
        st.subheader("🩺 Health Metrics")
        bmi = st.slider("BMI", 15.0, 50.0, 28.5, 0.1)
        waist_hip = st.slider("Waist-to-Hip Ratio", 0.6, 1.2, 0.85, 0.01)
        systolic_bp = st.slider("Systolic BP", 80, 200, 130)
        diastolic_bp = st.slider("Diastolic BP", 50, 130, 85)
        heart_rate = st.slider("Heart Rate", 40, 120, 75)
        cholesterol = st.slider("Total Cholesterol", 100, 350, 200)
        hdl = st.slider("HDL Cholesterol", 20, 100, 45)
        ldl = st.slider("LDL Cholesterol", 50, 250, 120)
        triglycerides = st.slider("Triglycerides", 50, 400, 150)
        
        st.subheader("📝 Medical History")
        family_history = st.checkbox("Family History of Diabetes")
        hypertension = st.checkbox("Hypertension History")
        cardiovascular = st.checkbox("Cardiovascular History")
    
    st.markdown("---")
    
    # Predict button
    if st.button("🔍 Predict Diabetes Risk", use_container_width=True):
        # Create progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i in range(100):
            progress_bar.progress(i + 1)
            if i < 30:
                status_text.text("Analyzing patient data...")
            elif i < 60:
                status_text.text("Processing health metrics...")
            elif i < 90:
                status_text.text("Calculating risk probability...")
            else:
                status_text.text("Generating results...")
            time.sleep(0.01)
        
        progress_bar.empty()
        status_text.empty()
        
        # Prepare input data
        patient_data = {
            'age': age,
            'alcohol_consumption_per_week': alcohol,
            'physical_activity_minutes_per_week': physical_activity,
            'diet_score': diet_score,
            'sleep_hours_per_day': sleep,
            'screen_time_hours_per_day': screen_time,
            'bmi': bmi,
            'waist_to_hip_ratio': waist_hip,
            'systolic_bp': systolic_bp,
            'diastolic_bp': diastolic_bp,
            'heart_rate': heart_rate,
            'cholesterol_total': cholesterol,
            'hdl_cholesterol': hdl,
            'ldl_cholesterol': ldl,
            'triglycerides': triglycerides,
            'gender': gender,
            'ethnicity': ethnicity,
            'education_level': education,
            'income_level': income,
            'smoking_status': smoking,
            'employment_status': employment,
            'family_history_diabetes': 1 if family_history else 0,
            'hypertension_history': 1 if hypertension else 0,
            'cardiovascular_history': 1 if cardiovascular else 0
        }
        
        # Create DataFrame
        test_df = pd.DataFrame([patient_data])
        test_df = test_df.reindex(columns=feature_columns, fill_value=0)
        
        # Encode categorical variables
        categorical_cols = ['gender', 'ethnicity', 'education_level', 'income_level', 
                           'smoking_status', 'employment_status']
        for col in categorical_cols:
            if col in test_df.columns:
                try:
                    test_df[col] = label_encoders[col].transform([str(test_df[col].iloc[0])])
                except:
                    test_df[col] = 0
        
        # Make prediction
        probability = model.predict_proba(test_df)[0, 1]
        
        # Display results
        st.markdown("---")
        st.markdown("## 📊 Prediction Results")
        
        # Risk level
        if probability > 0.7:
            risk_level = "HIGH RISK"
            risk_class = "risk-high"
            emoji = "🔴"
        elif probability > 0.4:
            risk_level = "MODERATE RISK"
            risk_class = "risk-moderate"
            emoji = "🟡"
        else:
            risk_level = "LOW RISK"
            risk_class = "risk-low"
            emoji = "🟢"
        
        # Display risk
        st.markdown(f'<div class="{risk_class}">{emoji} {risk_level}<br>Probability: {probability:.1%}</div>', 
                   unsafe_allow_html=True)
        
        # Metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Risk Probability", f"{probability:.1%}")
        with col2:
            st.metric("Risk Category", risk_level)
        with col3:
            confidence = "High" if probability > 0.7 or probability < 0.3 else "Moderate"
            st.metric("Confidence", confidence)
        
        # Recommendations
        st.markdown("---")
        st.markdown("## 💡 Recommendations")
        
        if probability > 0.5:
            st.warning("⚠️ **High Risk Detected** - Please consult a healthcare professional immediately.")
            st.markdown("""
            **Suggested Actions:**
            - 🏥 Schedule an appointment with your doctor
            - 🩸 Get blood glucose levels tested
            - 🥗 Improve diet and reduce sugar intake
            - 🏃 Increase physical activity
            - ⚖️ Work on weight management if BMI is high
            """)
        else:
            st.success("✅ **Low Risk** - Keep maintaining a healthy lifestyle!")
            st.markdown("""
            **Continue Good Habits:**
            - 🥗 Maintain a balanced diet
            - 🏃 Stay physically active
            - 😴 Get adequate sleep
            - 🚭 Avoid smoking and excessive alcohol
            - 🩺 Regular health check-ups
            """)
        
        # Risk factors
        st.markdown("---")
        st.markdown("## 🎯 Key Risk Factors")
        
        risk_factors = []
        if bmi > 30:
            risk_factors.append("⚠️ High BMI (Obesity)")
        if systolic_bp > 140:
            risk_factors.append("⚠️ High Blood Pressure")
        if family_history:
            risk_factors.append("⚠️ Family History of Diabetes")
        if physical_activity < 150:
            risk_factors.append("⚠️ Low Physical Activity")
        if age > 45:
            risk_factors.append("⚠️ Age Factor")
        
        if risk_factors:
            for factor in risk_factors:
                st.markdown(f"- {factor}")
        else:
            st.success("✅ No major risk factors detected!")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>⚕️ This is a predictive tool and should not replace professional medical advice.</p>
    <p>Always consult with healthcare professionals for accurate diagnosis and treatment.</p>
</div>
""", unsafe_allow_html=True)