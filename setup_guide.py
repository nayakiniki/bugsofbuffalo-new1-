#!/usr/bin/env python3
"""
Setup Guide for Cattle Breed Recognition Project
Run this script for step-by-step instructions
"""

import os
import subprocess
import sys

def run_command(command, description):
    print(f"\n🔧 {description}")
    print(f"   Running: {command}")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print("   ✅ Success")
        return True
    except subprocess.CalledProcessError as e:
        print(f"   ❌ Failed: {e}")
        return False

def print_step(step_number, description):
    print(f"\n{'='*60}")
    print(f"STEP {step_number}: {description}")
    print(f"{'='*60}")

def main():
    print("🐄 Cattle Breed Recognition Project Setup Guide 🐄")
    print("This script will guide you through the setup process.")
    
    # Step 1: Virtual Environment
    print_step(1, "Setting up Virtual Environment")
    if not os.path.exists('venv'):
        run_command("python -m venv venv", "Creating virtual environment")
    else:
        print("   ✅ Virtual environment already exists")
    
    # Step 2: Activate venv and install requirements
    print_step(2, "Installing Dependencies")
    
    # Determine the activation command based on OS
    if sys.platform == "win32":
        activate_cmd = "venv\\Scripts\\activate"
        pip_cmd = "venv\\Scripts\\pip"
    else:
        activate_cmd = "source venv/bin/activate"
        pip_cmd = "venv/bin/pip"
    
    # Install ML requirements
    ml_reqs = run_command(f"{pip_cmd} install -r ml-model/requirements.txt", "Installing ML dependencies")
    
    # Install Streamlit requirements
    streamlit_reqs = run_command(f"{pip_cmd} install -r streamlit-app/requirements.txt", "Installing Streamlit dependencies")
    
    # Step 3: Dataset setup
    print_step(3, "Setting up Dataset")
    print("Please choose one of the following options:")
    print("1. Auto-download (if available)")
    print("2. Manual download from Kaggle")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "1":
        run_command("python download_data.py", "Attempting auto-download")
    else:
        print("\n📥 Manual Download Instructions:")
        print("1. Visit: https://www.kaggle.com/datasets/lukex9442/indian-bovine-breeds")
        print("2. Click 'Download' (requires Kaggle account)")
        print("3. Extract the zip file to 'ml-model/data/indian-bovine-breeds/'")
        print("4. The folder structure should be:")
        print("   ml-model/data/indian-bovine-breeds/train/")
        print("   ml-model/data/indian-bovine-breeds/validation/")
        print("   ml-model/data/indian-bovine-breeds/test/")
    
    # Step 4: Training instructions
    print_step(4, "Training the Model")
    print("To train the model, run:")
    print("   cd ml-model")
    if sys.platform == "win32":
        print("   ..\\venv\\Scripts\\python train.py")
    else:
        print("   ../venv/bin/python train.py")
    
    # Step 5: Running the app
    print_step(5, "Running the Streamlit App")
    print("To run the web application:")
    print("   cd streamlit-app")
    if sys.platform == "win32":
        print("   ..\\venv\\Scripts\\streamlit run app.py")
    else:
        print("   ../venv/bin/streamlit run app.py")
    
    print("\n🎉 Setup complete! Follow the instructions above to continue.")

if __name__ == "__main__":
    main()
