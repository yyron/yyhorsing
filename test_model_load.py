from tensorflow.keras.models import load_model
import os

# Define the path to the re-saved model
model_path = "/Users/cy/streamlit_project/models/model.kerasCOLD2025MARCHSMOTnewformat.keras"

# Check if the file exists
if not os.path.exists(model_path):
    print(f"Model file not found at: {model_path}")
else:
    print(f"Model file found at: {model_path}")

# Attempt to load the re-saved model
try:
    model = load_model(model_path)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading the model: {e}")