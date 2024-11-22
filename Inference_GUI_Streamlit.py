import os
import sys
import streamlit as st
import pandas as pd
from inference import load_model, preprocess_input, make_prediction

# Add the root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Create necessary folders if they don't exist
os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

# Load existing models dynamically from the models/ folder
model_files = {f.split(".")[0]: os.path.join("models", f) for f in os.listdir("models") if f.endswith(".pkl")}
models = {name: load_model(path) for name, path in model_files.items()}

st.title("Enhanced Inference GUI")

# Sidebar for uploading new models
st.sidebar.header("Upload New Models")
uploaded_model = st.sidebar.file_uploader("Upload a New Model", type=["pkl"])
if uploaded_model:
    model_save_path = os.path.join("models", uploaded_model.name)
    with open(model_save_path, "wb") as f:
        f.write(uploaded_model.read())
    st.sidebar.success(f"Model saved to {model_save_path}")
    # Refresh the models list
    model_files = {f.split(".")[0]: os.path.join("models", f) for f in os.listdir("models") if f.endswith(".pkl")}
    models = {name: load_model(path) for name, path in model_files.items()}

# Radio button to select the prediction task
task_choice = st.radio("Select the Prediction Task:", list(models.keys()))
selected_model = models[task_choice]

# Common features for all models
features = [
    "Daily_Time_Spent_on_Site",
    "Age",
    "Area_Income",
    "Daily_Internet_Usage"
]

# Dropdown to select input method
input_method = st.radio("Choose Input Method:", ["Manual Input", "Upload CSV"])

if input_method == "Manual Input":
    st.subheader("Manual Input")
    manual_input = {}
    for feature in features:
        manual_input[feature] = st.number_input(f"Enter {feature}:", min_value=0.0)

    if st.button("Predict from Manual Input"):
        manual_input_df = pd.DataFrame([manual_input])
        processed_manual_data = preprocess_input(manual_input_df, features)
        manual_prediction, manual_probabilities = make_prediction(
            selected_model, processed_manual_data
        )
        st.subheader("Manual Prediction Result")
        st.write(f"Prediction for {task_choice}: {manual_prediction[0]}")
        if manual_probabilities is not None:
            st.write(f"Probabilities: {manual_probabilities[0]}")

elif input_method == "Upload CSV":
    st.subheader("Upload CSV for Batch Predictions")
    uploaded_file = st.file_uploader("Upload your input CSV", type=["csv"])
    if uploaded_file:
        uploaded_data = pd.read_csv(uploaded_file)
        st.write("Uploaded Data:")
        st.dataframe(uploaded_data)

        if st.button("Predict from CSV"):
            processed_data = preprocess_input(uploaded_data, features)
            predictions, probabilities = make_prediction(selected_model, processed_data)

            # Append predictions and probabilities to the data
            uploaded_data["Prediction"] = predictions
            if probabilities is not None:
                for i in range(probabilities.shape[1]):
                    uploaded_data[f"Probability_{i}"] = probabilities[:, i]

            st.subheader(f"Results for {task_choice}")
            st.dataframe(uploaded_data)

            # Save results to the results folder
            result_file_path = os.path.join("results", f"{task_choice.replace(' ', '_').lower()}_predictions.csv")
            uploaded_data.to_csv(result_file_path, index=False)

            # Allow download of the updated CSV
            csv = uploaded_data.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download Results as CSV",
                data=csv,
                file_name=f"{task_choice.replace(' ', '_').lower()}_predictions.csv",
                mime="text/csv",
            )
            st.success(f"Results saved to: {result_file_path}")

