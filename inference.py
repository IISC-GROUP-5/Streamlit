import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler
import json


def load_model(model_file):
    """
    Load the trained model from a .pkl file.
    """
    with open(model_file, 'rb') as file:
        model = pickle.load(file)
    print(f"Model loaded successfully from {model_file}")
    return model


def preprocess_input(input_data, feature_columns, scaler=None):
    """
    Preprocess the input data to match the format expected by the model,
    applying standardization if a scaler is provided.
    """
    if isinstance(input_data, dict):
        input_df = pd.DataFrame([input_data])  # Single instance as a DataFrame
    elif isinstance(input_data, list):
        input_df = pd.DataFrame(input_data)  # List of instances
    elif isinstance(input_data, pd.DataFrame):
        input_df = input_data
    else:
        raise ValueError("Unsupported input data format. Use dict, list, or Pandas DataFrame.")

    # Replace spaces in column names with underscores
    input_df.columns = input_df.columns.str.replace(' ', '_')
       # Handle 'Age' column specifically: Ensure absolute values and handle missing data
    if 'Age' in input_df.columns:
        input_df['Age'] = input_df['Age'].abs()  # Ensure 'Age' is absolute
        if input_df['Age'].isnull().sum() > 0:  # If there are missing values
            if abs(input_df['Age'].skew()) >= 0.5:  # Check for skewness
                # Replace with absolute median
                median_age = abs(input_df['Age'].median())
                input_df['Age'] = input_df['Age'].fillna(value=median_age)
            else:
                # Replace with absolute mean
                mean_age = abs(input_df['Age'].mean())
                input_df['Age'] = input_df['Age'].fillna(value=mean_age)

    # Handle missing values for other columns
    for column in input_df.columns:
        if column != 'Age':  # 'Age' already handled
            if input_df[column].dtype == 'object':  # Categorical data
                most_frequent = input_df[column].mode()[0]
                input_df[column] = input_df[column].fillna(value=most_frequent)
            else:  # Numeric data
                if input_df[column].isnull().sum() > 0:  # Only process columns with missing values
                    if abs(input_df[column].skew()) >= 0.5:  # Check for skewness
                        # Replace with absolute median
                        median_value = abs(input_df[column].median())
                        input_df[column] = input_df[column].fillna(value=median_value)
                    else:
                        # Replace with absolute mean
                        mean_value = abs(input_df[column].mean())
                        input_df[column] = input_df[column].fillna(value=mean_value)
    # Ensure only the required columns are used
    input_df = input_df[feature_columns]

    if scaler:
        # Standardize input data using the provided scaler
        input_df_scaled = pd.DataFrame(scaler.transform(input_df), columns=feature_columns)
        return input_df_scaled
    else:
        # Return raw data if no scaling is applied
        return input_df


def make_prediction(model, input_data):
    """
    Generate predictions and probabilities using the loaded model.
    """
    predictions = model.predict(input_data)
    probabilities = model.predict_proba(input_data) if hasattr(model, 'predict_proba') else None
    return predictions, probabilities


def save_inference_results(output_file, input_data, predictions, probabilities, original_data):
    """
    Save the inference results to a file in JSON format.
    The `original_data` is used to store input in the original form.
    """
    results = []
    for i, row in enumerate(original_data.to_dict(orient='records')):
        result = {
            "input_data": row,
            "prediction": int(predictions[i])
        }
        if probabilities is not None:
            result["probabilities"] = list(probabilities[i])
        results.append(result)

    # Save to a JSON file
    with open(output_file, 'w') as file:
        json.dump(results, file, indent=4)

    print(f"Inference results saved to {output_file}")


# Main inference logic
if __name__ == "__main__":
    # List of model file paths
    model_files = [
        "naive_bayes_Gender_model.pkl",
        "naive_bayes_Ad_click_model.pkl",
        "svm_Gender_model.pkl",
        "svm_Ad_click_model.pkl"
    ]

    # Define feature columns (must match the training data)
    feature_columns = ['Daily_Time_Spent_on_Site', 'Age', 'Area_Income', 'Daily_Internet_Usage']

    # Input data for prediction (example)
    example_input = pd.read_csv("Test_Data.csv")

    # Loop through each model file
    for model_file in model_files:
        # Load the trained model
        model = load_model(model_file)

        # Preprocess the input (scale the input data)
        # scaler = StandardScaler()  # You can use a previously fitted scaler if available
        input_data_scaled = preprocess_input(example_input, feature_columns)

        # Make predictions
        predictions, probabilities = make_prediction(model, input_data_scaled)

        # Save results to a JSON file
        base_name = model_file.rsplit('.', 1)[0]  # Remove extension
        output_file = f"{base_name}_inference_results.json"

        # Save inference results in the original input data format
        original_data = pd.DataFrame(example_input)  # Create a DataFrame with original data
        save_inference_results(output_file, input_data_scaled, predictions, probabilities, original_data)
