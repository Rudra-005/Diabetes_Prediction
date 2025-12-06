#!/usr/bin/env python3
"""
Setup script for Diabetes Prediction Challenge
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    print("Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✓ All packages installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"✗ Error installing packages: {e}")
        return False
    return True

def check_data_files():
    """Check if required data files exist"""
    required_files = ['train.csv', 'test.csv', 'sample_submission.csv']
    missing_files = []
    
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(f"✗ Missing data files: {', '.join(missing_files)}")
        print("Please download the competition data from Kaggle and place it in this directory.")
        return False
    else:
        print("✓ All required data files found!")
        return True

def main():
    """Main setup function"""
    print("=" * 60)
    print("DIABETES PREDICTION CHALLENGE SETUP")
    print("=" * 60)
    
    # Check data files
    if not check_data_files():
        return
    
    # Install requirements
    if not install_requirements():
        return
    
    print("\n" + "=" * 60)
    print("SETUP COMPLETE!")
    print("=" * 60)
    print("You can now run the analysis in one of two ways:")
    print("1. Jupyter Notebook: jupyter notebook diabetes_prediction.ipynb")
    print("2. Python Script: python diabetes_prediction.py")
    print("=" * 60)

if __name__ == "__main__":
    main()