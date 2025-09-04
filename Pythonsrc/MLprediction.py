import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from ML import load_ml_data

def predict_with_model(csv_file_path, modelname, base_path, output_csv_path=None):
    """
    Load a saved model and make predictions on new CSV data.
    
    Parameters:
    csv_file_path (str): Path to the CSV file containing new data for predictions
    modelname (str): Name of the saved model (without _model suffix)
    base_path (str): Base path where MLmodels directory is located
    output_csv_path (str): Path to save predictions CSV (optional, will auto-generate if None)
    
    Returns:
    str: Path to the saved predictions CSV file
    """
    
    print(f"=== Making Predictions with Model: {modelname} ===")
    
    # Construct paths to model files
    model_dir = os.path.join(base_path, "MLmodels", f"{modelname}_model")
    metadata_path = os.path.join(model_dir, f"{modelname}_metadata.json")
    model_path = os.path.join(model_dir, f"{modelname}.keras")

    # Check if model directory and files exist
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Model metadata not found: {metadata_path}")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Load model metadata to get input/output feature names
    print("Loading model metadata...")
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    input_features = metadata['data_info']['input_features']
    output_features = metadata['data_info']['output_features']
    
    print(f"Model input features ({len(input_features)}): {input_features}")
    print(f"Model output features ({len(output_features)}): {output_features}")
    
    # Load the trained model
    print("Loading trained model...")
    model = tf.keras.models.load_model(model_path)
    print("Model loaded successfully!")
    
    # Load and prepare the new data using the same features as the model
    print(f"Loading prediction data from: {csv_file_path}")
    
    # Create selected_columns list that includes both input and output features
    # (we need output features in case they exist in the data, but we won't use them for prediction)
    all_model_features = input_features + output_features
    
    try:
        identifiers, feature_names, data_array, headers = load_ml_data(
            csv_file_path=csv_file_path,
            selected_columns=all_model_features,
            filter_by_column=None,  # Don't filter rows for prediction
            fill_nan_with_median=True  # Fill NaN values for prediction
        )
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Trying to load with only input features...")
        # Fallback: try loading with only input features
        identifiers, feature_names, data_array, headers = load_ml_data(
            csv_file_path=csv_file_path,
            selected_columns=input_features,
            filter_by_column=None,
            fill_nan_with_median=True
        )
    
    # Extract only the input features for prediction
    input_indices = []
    for feat in input_features:
        if feat in feature_names:
            input_indices.append(feature_names.index(feat))
        else:
            raise ValueError(f"Required input feature '{feat}' not found in the CSV data")
    
    # Prepare input data for prediction
    X_pred = data_array[:, input_indices]
    
    print(f"Prediction input shape: {X_pred.shape}")
    print(f"Number of samples to predict: {len(identifiers)}")
    
    # Make predictions
    print("Making predictions...")
    predictions = model.predict(X_pred, verbose=0)
    
    print(f"Predictions shape: {predictions.shape}")
    print("Predictions completed successfully!")
    
    # Prepare results for CSV output
    results_data = {'identifier': identifiers}
    
    # Add predicted values for each output feature
    if len(output_features) == 1:
        # Single output
        results_data[f'predicted_{output_features[0]}'] = predictions.flatten()
    else:
        # Multiple outputs
        for i, output_feature in enumerate(output_features):
            results_data[f'predicted_{output_feature}'] = predictions[:, i]
    
    # Create results DataFrame
    results_df = pd.DataFrame(results_data)
    
    # Generate output path if not provided
    if output_csv_path is None:
        csv_dir = os.path.dirname(csv_file_path)
        csv_filename = os.path.splitext(os.path.basename(csv_file_path))[0]
        output_csv_path = os.path.join(csv_dir, f"{csv_filename}_{modelname}_predictions.csv")
    
    # Save predictions to CSV
    results_df.to_csv(output_csv_path, index=False)
    print(f"Predictions saved to: {output_csv_path}")
    
    # Print sample predictions
    print(f"\nSample predictions (first 5 rows):")
    print(results_df.head())

    return output_csv_path

if __name__ == "__main__":
    # Set your paths here
    base_path = "/Users/oliverreed/Library/CloudStorage/GoogleDrive-or280@cam.ac.uk/Shared drives/MET - ShapeMemoryAlloyResearch/Useful/Python Scripts/MLFiles"
    modelname = 'Ms_prediction'
    csv_file_path = os.path.join(base_path, "ML_Database.csv")  # Change this to your new data file
    
    try:
        output_path = predict_with_model(
            csv_file_path=csv_file_path,
            modelname=modelname,
            base_path=base_path
        )
        
    except Exception as e:
        print(f"Error during prediction: {e}")
