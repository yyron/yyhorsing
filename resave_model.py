from tensorflow.keras.models import Sequential

# Create a dummy model (replace this with your actual model if needed)
model = Sequential()

# Save the model as a valid `.keras` file
model.save("models/model.kerasCOLD2025MARCHSMOTnewformat.keras", save_format="keras")
print("Model re-saved successfully!")