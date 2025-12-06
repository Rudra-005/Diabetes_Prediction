# Diabetes Prediction Challenge - Kaggle Playground Series S5E12

This project implements a machine learning solution for predicting diabetes diagnosis probability using patient health data.

## Competition Overview
- **Goal**: Predict the probability that a patient will be diagnosed with diabetes
- **Evaluation**: Area under the ROC curve (AUC-ROC)
- **Data**: Synthetically generated from real-world data

## Features
The dataset includes the following features:
- **Demographics**: age, gender, ethnicity, education_level, income_level
- **Lifestyle**: alcohol_consumption_per_week, physical_activity_minutes_per_week, diet_score, sleep_hours_per_day, screen_time_hours_per_day, smoking_status, employment_status
- **Health Metrics**: bmi, waist_to_hip_ratio, systolic_bp, diastolic_bp, heart_rate
- **Lab Results**: cholesterol_total, hdl_cholesterol, ldl_cholesterol, triglycerides
- **Medical History**: family_history_diabetes, hypertension_history, cardiovascular_history

## Solution Approach
1. **Data Preprocessing**:
   - Label encoding for categorical variables
   - Feature scaling for logistic regression
   - Consistent encoding across train/test sets

2. **Model Training**:
   - Random Forest Classifier
   - Gradient Boosting Classifier
   - Logistic Regression
   - Cross-validation for model selection

3. **Model Selection**:
   - Best model chosen based on ROC-AUC score
   - Validation split for unbiased evaluation

## Usage

### Quick Setup (Windows)
```bash
run_setup.bat
```

### Manual Setup
1. **Install Dependencies**
```bash
pip install -r requirements.txt
```

2. **Run Analysis**

**Option A: Jupyter Notebook (Recommended)**
```bash
jupyter notebook diabetes_prediction.ipynb
```

**Option B: Python Script**
```bash
python diabetes_prediction.py
```

**Option C: Setup Script**
```bash
python setup.py
```

### What the analysis does:
1. **Exploratory Data Analysis**: Visualize data distributions and relationships
2. **Data Preprocessing**: Handle categorical variables and feature scaling
3. **Model Training**: Train Random Forest, Gradient Boosting, and Logistic Regression
4. **Model Evaluation**: Compare models using ROC-AUC score
5. **Feature Analysis**: Identify most important features
6. **Prediction Generation**: Create submission file with test predictions

## Files
- `diabetes_prediction.ipynb`: **Main Jupyter notebook with complete analysis**
- `diabetes_prediction.py`: Python script version
- `requirements.txt`: Python dependencies
- `setup.py`: Setup and verification script
- `run_setup.bat`: Windows batch setup file
- `train.csv`: Training data (download from Kaggle)
- `test.csv`: Test data (download from Kaggle)
- `sample_submission.csv`: Sample submission format (download from Kaggle)
- `submission.csv`: Generated predictions (created after running analysis)
- `README.md`: This file

## Model Performance
The script automatically evaluates multiple models and selects the best one based on ROC-AUC score on validation data.

## Submission Format
The output file `submission.csv` contains:
- `id`: Test sample identifier
- `diagnosed_diabetes`: Predicted probability (0-1)