import pickle
import pandas as pd
import numpy as np
import os  # <--- Import os module

# --- FIX: Get the folder where THIS script is located ---
script_dir = os.path.dirname(os.path.abspath(__file__))

# --- Construct absolute paths to the files ---
model_filename = os.path.join(script_dir, 'house_price_model.pkl')
scaler_filename = os.path.join(script_dir, 'scaler.pkl')

print(f"Looking for model at: {model_filename}") # Debug print

try:
    with open(model_filename, 'rb') as f:
        model = pickle.load(f)
    
    with open(scaler_filename, 'rb') as f:
        scaler = pickle.load(f)
        
    print("✅ Model and Scaler loaded successfully.\n")

except FileNotFoundError:
    print(f"❌ Error: Could not find the files.")
    print(f"Make sure 'house_price_model.pkl' is in this folder: {script_dir}")
    exit()

# ... (The rest of your code: Input Data, DataFrame, etc.) ...