import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow import keras


class CustomEarlyStopping(keras.callbacks.Callback):
    """
    Custom early stopping callback with specific conditions:
    1. Stop if validation loss increases to 10% above its minimum
    2. Stop if validation loss becomes twice as large as training loss
    3. Stop at max epochs (handled by fit() max_epochs parameter)
    """
    def __init__(self, patience=10, verbose=0):
        super(CustomEarlyStopping, self).__init__()
        self.patience = patience
        self.verbose = verbose
        self.best_val_loss = float('inf')
        self.wait = 0

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}

        current_val_loss = logs.get('val_loss')
        current_train_loss = logs.get('loss')

        if current_val_loss is None or current_train_loss is None:
            return

        # Update best validation loss
        if current_val_loss < self.best_val_loss:
            self.best_val_loss = current_val_loss
            self.wait = 0
        else:
            self.wait += 1

        # Condition 1: Stop if validation loss is 10% above its minimum
        if current_val_loss > self.best_val_loss * 1.1:
            if self.verbose > 0:
                print(f"\nEarly stopping: Validation loss ({current_val_loss:.4f}) is 10% above minimum ({self.best_val_loss:.4f})")
            self.model.stop_training = True
            return

        # Condition 2: Stop if validation loss is twice the training loss
        if current_val_loss > current_train_loss * 10.0:
            if self.verbose > 0:
                print(f"\nEarly stopping: Validation loss ({current_val_loss:.4f}) is twice training loss ({current_train_loss:.4f})")
            self.model.stop_training = True
            return
        

def load_ml_data(csv_file_path, selected_columns=None, filter_by_column=None, fill_nan_with_median=False):
    """
    Comprehensive function to load and prepare CSV data for machine learning.

    Parameters:
    csv_file_path (str): Full path to the CSV file (can be any CSV file with ML data)
    selected_columns (list): List of column prefixes to select (e.g., ['D1', 'D2', 'E5', 'P1'])
                           If None, all columns are returned
    filter_by_column (str): Column prefix to filter rows by (e.g., 'P1'). Only rows with 
                          non-empty values in this column will be kept. If None, all rows are kept.
    fill_nan_with_median (bool): If True, fill NaN values with column median; if False, remove rows with NaN

    Returns:
    tuple: (identifiers, feature_names, data_array, headers)
        - identifiers: array of identifier values from first column
        - feature_names: list of column names (excluding identifier column)
        - data_array: 2D numpy array of numerical data (excluding identifier column)
        - headers: list of all column headers including identifier
    """

    # Check if file exists
    if not os.path.exists(csv_file_path):
        raise FileNotFoundError(f"CSV file not found: {csv_file_path}")

    # Read the CSV file
    df = pd.read_csv(csv_file_path)

    # Get all headers
    headers = df.columns.tolist()

    print(f"Loaded data from {csv_file_path}")
    print(f"Original dataset shape: {df.shape}")

    # Filter rows based on specified column if requested
    if filter_by_column is not None:
        # Find the full column name that matches the filter prefix
        filter_col_name = None
        for col in headers[1:]:  # Skip identifier column
            if col.startswith(filter_by_column + '_') or col == filter_by_column:
                filter_col_name = col
                break

        if filter_col_name is None:
            raise ValueError(f"No column found for filter prefix '{filter_by_column}'")

        # Filter rows where the specified column has non-empty values
        # Handle both NaN and empty string cases
        before_filter_count = len(df)
        df = df.dropna(subset=[filter_col_name])  # Remove NaN values
        df = df[df[filter_col_name] != '']  # Remove empty strings
        after_filter_count = len(df)

        print(f"Filtered by column '{filter_col_name}': {before_filter_count} -> {after_filter_count} rows")

        if len(df) == 0:
            raise ValueError(f"No rows remaining after filtering by column '{filter_by_column}'")

    # Extract identifiers (first column) after filtering
    identifiers = df.iloc[:, 0].values

    # If specific columns are selected, filter the dataframe
    if selected_columns is not None:
        # Find matching column names for the selected prefixes
        selected_full_names = []
        for prefix in selected_columns:
            # Find columns that start with the prefix
            matching_cols = [col for col in headers[1:] if col.startswith(prefix + '_') or col == prefix]
            if matching_cols:
                selected_full_names.extend(matching_cols)
            else:
                print(f"Warning: No columns found for prefix '{prefix}'")

        if not selected_full_names:
            raise ValueError("No valid columns found for the specified prefixes")

        # Extract selected feature names and data
        feature_names = selected_full_names
        data_array = df[selected_full_names].values

        print(f"Selected columns: {selected_columns}")
        
    else:
        # Extract all feature names (all columns except first)
        feature_names = headers[1:]
        # Extract numerical data (all columns except first)
        data_array = df.iloc[:, 1:].values

    # Remove columns where all values are identical (constant columns)
    if len(data_array) > 1:  # Only check if we have more than one sample
        # Convert to pandas DataFrame for easier manipulation
        temp_df = pd.DataFrame(data_array, columns=feature_names)

        # Find columns with constant values (including NaN handling)
        constant_columns = []
        for col in temp_df.columns:
            # Get non-NaN values for this column
            non_nan_values = temp_df[col].dropna()
            if len(non_nan_values) > 0:
                # Check if all non-NaN values are the same
                unique_values = non_nan_values.nunique()
                if unique_values <= 1:
                    constant_columns.append(col)

        if constant_columns:
            print(f"Removing {len(constant_columns)} constant columns: {constant_columns}")

            # Remove constant columns
            remaining_columns = [col for col in feature_names if col not in constant_columns]
            if remaining_columns:
                feature_names = remaining_columns
                data_array = temp_df[remaining_columns].values
            else:
                raise ValueError("All columns have constant values - no features remaining for analysis")
        else:
            print("No constant columns found - all features have variation")

    # Convert data to numeric and handle NaN values
    temp_df = pd.DataFrame(data_array, columns=feature_names)
    temp_df = temp_df.apply(pd.to_numeric, errors='coerce')

    # Check for NaN values and handle them
    nan_mask = temp_df.isnull()
    rows_with_nan = nan_mask.any(axis=1)

    if rows_with_nan.any():
        if fill_nan_with_median:
            # Calculate medians for each column
            medians = temp_df.median()

            # Fill NaN values with medians
            temp_df = temp_df.fillna(medians)

            # Update identifiers and data_array (no rows removed)
            data_array = temp_df.values
            print(f"Filled {nan_mask.sum().sum()} NaN values with column medians")
        else:
            print(f"Removing {rows_with_nan.sum()} rows with NaN values...")
            # Remove rows with NaN values
            temp_df = temp_df[~rows_with_nan]
            identifiers = identifiers[~rows_with_nan]
            data_array = temp_df.values
    else:
        print("No NaN values found in the data")
        data_array = temp_df.values

    print(f"Final data shape: {data_array.shape}")
    print(f"Features: {feature_names}")
    print(f"Number of samples: {len(identifiers)}")

    return identifiers, feature_names, data_array, headers


def train_test_split(X, y, test_size=0.2, random_state=None, shuffle=True):
    """
    Manual implementation of train_test_split to avoid sklearn dependency.

    Parameters:
    X: Features array
    y: Target array  
    test_size: Proportion of data to use for testing (default 0.2)
    random_state: Random seed for reproducibility
    shuffle: Whether to shuffle data before splitting

    Returns:
    X_train, X_test, y_train, y_test
    """
    if random_state is not None:
        np.random.seed(random_state)

    n_samples = len(X)
    n_test = int(n_samples * test_size)

    # Create indices
    indices = np.arange(n_samples)

    if shuffle:
        np.random.shuffle(indices)

    # Split indices
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]

    # Split data
    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]

    return X_train, X_test, y_train, y_test


def build_and_train_neural_network(data_array, feature_names, identifiers=None, e_columns_as_inputs=True,
                                   epochs=100, validation_split=0.2, verbose=1, batch_size=16,
                                   first_layer_nodes=16, second_layer_nodes=8, reserve_samples=True):
    """
    Build, compile and train a neural network using the materials database.

    Parameters:
    data_array (np.array): 2D numpy array with the data
    feature_names (list): List of feature column names
    identifiers (np.array): Array of sample identifiers (optional)
    e_columns_as_inputs (bool): If True, treat E columns as inputs; if False, treat as outputs
    epochs (int): Number of training epochs
    validation_split (float): Fraction of data to use for validation
    verbose (int): Verbosity level for training
    batch_size (int): Batch size for training
    first_layer_nodes (int): Number of nodes in first hidden layer
    second_layer_nodes (int): Number of nodes in second hidden layer
    reserve_samples (bool): Whether to reserve samples for final testing

    Returns:
    tuple: (model, history, input_features, output_features)
        - model: Trained Keras model
        - history: Training history object
        - input_features: List of input feature names
        - output_features: List of output feature names
    """

    # Separate input and output features based on column prefixes
    input_features = []
    output_features = []

    for feature in feature_names:
        if feature.startswith('D'):
            input_features.append(feature)
        elif feature.startswith('P'):
            output_features.append(feature)
        elif feature.startswith('E'):
            if e_columns_as_inputs:
                input_features.append(feature)
            else:
                output_features.append(feature)
        else:
            print(f"Warning: Unknown column prefix for '{feature}', treating as input")
            input_features.append(feature)

    if reserve_samples:
        print(f"Input features ({len(input_features)}): {input_features}")
        print(f"Output features ({len(output_features)}): {output_features}")

    if len(input_features) == 0:
        raise ValueError("No input features found")
    if len(output_features) == 0:
        raise ValueError("No output features found")

    # Get indices for input and output columns
    input_indices = [feature_names.index(feat) for feat in input_features]
    output_indices = [feature_names.index(feat) for feat in output_features]

    # Split data into inputs and outputs (data is already cleaned from load_ml_database)
    X = data_array[:, input_indices]
    y = data_array[:, output_indices]

    if reserve_samples:
        print(f"Original data shape: X={X.shape}, y={y.shape}")

    if len(X) == 0:
        raise ValueError("No training samples available")

    # Reserve 5 random samples for final testing (only if reserve_samples is True)
    X_reserved = None
    y_reserved = None
    identifiers_reserved = None

    if reserve_samples:
        if len(X) < 10:
            raise ValueError("Need at least 10 samples to reserve 5 for testing")

        # Set random seed for reproducibility
        np.random.seed(42)

        # Randomly select 5 indices
        test_indices = np.random.choice(len(X), size=5, replace=False)
        remaining_indices = np.setdiff1d(np.arange(len(X)), test_indices)

        # Extract test samples
        X_reserved = X[test_indices]
        y_reserved = y[test_indices]

        # Extract identifiers for reserved samples if available
        if identifiers is not None:
            identifiers_reserved = identifiers[test_indices]

        # Use remaining data for training
        X = X[remaining_indices]
        y = y[remaining_indices]

        print(f"Reserved 5 samples for final testing")
        print(f"Training data shape after reservation: X={X.shape}, y={y.shape}")
    else:
        # When not reserving samples, use all data for training
        print(f"Using all data for training: X={X.shape}, y={y.shape}")

    # Build the neural network model
    model = keras.Sequential([
        # Input layer (specify input shape)
        keras.Input(shape=(len(input_features),)),

        keras.layers.Normalization(),
        keras.layers.Dense(first_layer_nodes, activation='tanh'),
        keras.layers.Dense(second_layer_nodes, activation='tanh'),
        keras.layers.Dense(len(output_features))
    ])

    # Adapt the normalization layer to the training data
    model.layers[0].adapt(X)

    # Compile the model
    model.compile(
        optimizer='adam',
        loss='mse',  # Mean squared error for regression
        metrics=['mae']  # Mean absolute error as additional metric
    )

    # Print model summary only if reserve_samples is True
    if reserve_samples:
        print("\nModel Architecture:")
        model.summary()

    # Randomly split data into training and validation sets using our custom function
    if validation_split > 0:
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, 
            test_size=validation_split, 
            random_state=42,  # For reproducible results
            shuffle=True
        )
        if reserve_samples:
            print(f"Training set: {X_train.shape[0]} samples")
            print(f"Validation set: {X_val.shape[0]} samples")

        # Train the model with validation data
        if reserve_samples:
            print(f"\nTraining model for up to {epochs} epochs with early stopping...")
        else:
            print(f"Training up to {epochs} epochs with early stopping...")

        # Create custom early stopping callback
        early_stopping = CustomEarlyStopping(patience=10, verbose=1 if reserve_samples else 0)

        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,  # Cap at 1500 epochs
            verbose=verbose,
            batch_size=batch_size,
            callbacks=[early_stopping]
        )
    else:
        # Train without validation
        if reserve_samples:
            print(f"\nTraining model for up to {epochs} epochs (no validation, no early stopping)...")
        else:
            print(f"Training up to {epochs} epochs (no validation, no early stopping)...")
        history = model.fit(
            X, y,
            epochs=epochs,  # Cap at 1500 epochs
            verbose=verbose,
            batch_size=batch_size
        )

    # Print final training results
    final_loss = history.history['loss'][-1]
    if 'val_loss' in history.history:
        final_val_loss = history.history['val_loss'][-1]
        if reserve_samples:
            print(f"\nTraining completed!")
            print(f"Final training loss: {final_loss:.4f}")
            print(f"Final validation loss: {final_val_loss:.4f}")
        else:
            print(f"Training finished - Loss: {final_loss:.4f}, Val Loss: {final_val_loss:.4f}")
    else:
        if reserve_samples:
            print(f"\nTraining completed!")
            print(f"Final training loss: {final_loss:.4f}")
        else:
            print(f"Training finished - Loss: {final_loss:.4f}")
    
    if reserve_samples:
        # Test the model on the 5 reserved samples
        print(f"\n=== Testing on Reserved Samples ===")
        predictions = model.predict(X_reserved, verbose=0)
    
        print(f"Prediction vs Actual comparison:")
        for i in range(5):
            # Display identifier if available
            if identifiers_reserved is not None:
                print(f"\nSample {i+1} (ID: {identifiers_reserved[i]}):")
            else:
                print(f"\nSample {i+1}:")
    
            print(f"  Predicted: {predictions[i]}")
            print(f"  Actual:    {y_reserved[i]}")
    
            # Calculate absolute differences
            abs_diff = np.abs(predictions[i] - y_reserved[i])
            print(f"  Abs Diff:  {abs_diff}")
            
    return model, history, input_features, output_features


def tune_hyperparameters(data_array, feature_names, identifiers=None, e_columns_as_inputs=True, 
                        epochs=100, validation_split=0.2, verbose=0, save_path=None, modelname='model'):
    """
    Tune hyperparameters by testing random combinations and plotting results.

    Parameters:
    data_array (np.array): 2D numpy array with the data
    feature_names (list): List of feature column names
    identifiers (np.array): Array of sample identifiers (optional)
    e_columns_as_inputs (bool): If True, treat E columns as inputs; if False, treat as outputs
    epochs (int): Number of training epochs
    validation_split (float): Fraction of data to use for validation
    verbose (int): Verbosity level for training
    save_path (str): Path to save the plot (optional)
    modelname (str): Name to include in saved files

    Returns:
    list: List of tuples containing (hyperparams, min_val_loss, history)
    """

    # Define hyperparameter ranges
    batch_sizes = [4, 8, 16]
    first_layer_nodes = [32, 64, 128]

    print(f"=== Hyperparameter Tuning ===")
    print(f"Batch sizes: {batch_sizes}")
    print(f"First layer nodes: {first_layer_nodes}")
    print(f"Second layer nodes: automatically set to half of first layer")

    # Generate all possible combinations
    import itertools
    all_combinations = list(itertools.product(batch_sizes, first_layer_nodes))

    # Randomly select 9 combinations
    np.random.seed(42)  # For reproducibility
    selected_combinations = np.random.choice(len(all_combinations), size=min(9, len(all_combinations)), replace=False)
    hyperparams_to_test = [all_combinations[i] for i in selected_combinations]

    print(f"Testing {len(hyperparams_to_test)} random combinations:")
    for i, (batch_size, nodes1) in enumerate(hyperparams_to_test):
        nodes2 = max(1, nodes1 // 2)  # Ensure at least 1 node
        print(f"  {i+1}: batch_size={batch_size}, first_layer={nodes1}, second_layer={nodes2}")

    # Store results
    results = []

    # Test each combination
    for i, (batch_size, nodes1) in enumerate(hyperparams_to_test):
        nodes2 = max(1, nodes1 // 2)  # Calculate second layer as half of first layer (minimum 1)
        print(f"\n--- Testing combination {i+1}/9: batch_size={batch_size}, nodes1={nodes1}, nodes2={nodes2} ---")

        try:
            # Train model with current hyperparameters
            model, history, input_features, output_features = build_and_train_neural_network(
                data_array, feature_names, identifiers, 
                e_columns_as_inputs=e_columns_as_inputs,
                epochs=epochs,
                validation_split=validation_split,
                verbose=verbose,
                batch_size=batch_size, 
                first_layer_nodes=nodes1, 
                second_layer_nodes=nodes2,
                reserve_samples=False  # Don't reserve samples during hyperparameter tuning
            )

            # Get minimum validation loss and its epoch
            if 'val_loss' in history.history:
                min_val_loss = min(history.history['val_loss'])
                min_val_loss_epoch = history.history['val_loss'].index(min_val_loss) + 1  # +1 for 1-based indexing
            else:
                min_val_loss = None
                min_val_loss_epoch = None

            # Store results
            hyperparams = {'batch_size': batch_size, 'first_layer': nodes1, 'second_layer': nodes2}
            results.append((hyperparams, min_val_loss, min_val_loss_epoch, history))

            print(f"Minimum validation loss: {min_val_loss:.4f}" if min_val_loss else "No validation loss available")

        except Exception as e:
            print(f"Error with combination {i+1}: {e}")
            # Store failed result
            hyperparams = {'batch_size': batch_size, 'first_layer': nodes1, 'second_layer': nodes2}
            results.append((hyperparams, None, None, None))

    # Plot all loss curves in a 3x3 grid
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    fig.suptitle('Hyperparameter Tuning Results', fontsize=16, fontweight='bold')

    for i, (hyperparams, min_val_loss, min_val_loss_epoch, history) in enumerate(results):
        row = i // 3
        col = i % 3
        ax = axes[row, col]

        if history is not None:
            # Plot training and validation loss
            ax.plot(history.history['loss'], label='Training Loss', color='blue', linewidth=2)
            if 'val_loss' in history.history:
                ax.plot(history.history['val_loss'], label='Validation Loss', color='red', linewidth=2)

            ax.set_title(f"BS={hyperparams['batch_size']}, L1={hyperparams['first_layer']}\n"
                        f"Min Val Loss: {min_val_loss:.4f}" if min_val_loss else f"BS={hyperparams['batch_size']}, L1={hyperparams['first_layer']}", 
                        fontsize=10)
            ax.set_xlabel('Epoch', fontsize=9)
            ax.set_ylabel('Loss', fontsize=9)
            ax.set_ylim(0, 5000)  # Set y-axis limits
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        else:
            # Show failed result
            ax.text(0.5, 0.5, f"BS={hyperparams['batch_size']}\nL1={hyperparams['first_layer']}\n\nFAILED", 
                   ha='center', va='center', transform=ax.transAxes, fontsize=10)
            ax.set_title("Training Failed", fontsize=10, color='red')

    plt.tight_layout()

    # Save plot if path provided
    if save_path:
        plot_filename = os.path.join(save_path, f"{modelname}_hp_tune_results.png")
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"\nHyperparameter tuning plot saved to: {plot_filename}")

    # Show plot
    #plt.show()

    # Print summary results
    print(f"\n=== Hyperparameter Tuning Summary ===")
    print(f"{'#':<3} {'Batch Size':<10} {'Layer 1':<8} {'Layer 2':<8} {'Min Val Loss':<12}")
    print("-" * 50)

    # Sort by validation loss (successful runs first, then by loss value)
    sorted_results = sorted(results, key=lambda x: (x[1] is None, x[1] if x[1] is not None else float('inf')))

    for i, (hyperparams, min_val_loss, min_val_loss_epoch, history) in enumerate(sorted_results):
        loss_str = f"{min_val_loss:.6f}" if min_val_loss is not None else "FAILED"
        print(f"{i+1:<3} {hyperparams['batch_size']:<10} {hyperparams['first_layer']:<8} {hyperparams['second_layer']:<8} {loss_str:<12}")

    # Find and highlight best combination
    successful_results = [(h, l, e, hist) for h, l, e, hist in results if l is not None]
    if successful_results:
        best_hyperparams, best_loss, best_epoch, _ = min(successful_results, key=lambda x: x[1])
        print(f"\nBest combination:")
        print(f"  Batch Size: {best_hyperparams['batch_size']}")
        print(f"  First Layer: {best_hyperparams['first_layer']} nodes")
        print(f"  Second Layer: {best_hyperparams['second_layer']} nodes")
        print(f"  Minimum Validation Loss: {best_loss:.6f}")
        print(f"  Achieved at Epoch: {best_epoch}")
    else:
        print(f"\nNo successful training runs!")

    # Save results to CSV file
    if save_path:
        csv_filename = os.path.join(save_path, f"{modelname}_hp_tune_results.csv")

        # Check if CSV file exists
        file_exists = os.path.exists(csv_filename)

        # Prepare data for CSV
        csv_data = []
        for hyperparams, min_val_loss, min_val_loss_epoch, history in results:
            csv_data.append({
                'batch_size': hyperparams['batch_size'],
                'first_layer_nodes': hyperparams['first_layer'],
                'second_layer_nodes': hyperparams['second_layer'],
                'min_validation_loss': min_val_loss if min_val_loss is not None else 'FAILED',
                'min_loss_epoch': min_val_loss_epoch if min_val_loss_epoch is not None else 'FAILED'
            })

        # Create DataFrame
        df = pd.DataFrame(csv_data)

        # Write to CSV
        if file_exists:
            # Append to existing file without header
            df.to_csv(csv_filename, mode='a', header=False, index=False)
            print(f"Results appended to existing CSV: {csv_filename}")
        else:
            # Create new file with header
            df.to_csv(csv_filename, mode='w', header=True, index=False)
            print(f"Results saved to new CSV: {csv_filename}")

    return results


def plot_training_history(history, save_path, filename="training_history.png"):
    """
    Plot training and validation loss over epochs and save to file.
    
    Parameters:
    history: Keras training history object
    save_path (str): Directory path where to save the plot
    filename (str): Name of the plot file (default: "training_history.png")
    
    Returns:
    str: Full path to the saved plot file
    """
    # Create the plot
    plt.figure(figsize=(8, 6))

    # Plot loss
    plt.plot(history.history['loss'], label='Training Loss', color='blue', linewidth=2)
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='Validation Loss', color='red', linewidth=2)
    plt.title('Model Loss Over Epochs', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss (MSE)', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # Adjust layout
    plt.tight_layout()
    
    # Ensure the save directory exists
    os.makedirs(save_path, exist_ok=True)
    
    # Save the plot
    full_path = os.path.join(save_path, filename)
    plt.savefig(full_path, dpi=300, bbox_inches='tight')
    print(f"Training history plot saved to: {full_path}")
    
    # Display the plot
    #plt.show()
    
    # Clear the figure to free memory
    plt.clf()
    plt.close()
    
    return full_path

def train_and_save_model(data_array, feature_names, identifiers=None, modelname='model', 
                        base_path=None, e_columns_as_inputs=True, epochs=1500, 
                        validation_split=0.2, batch_size=16, first_layer_nodes=32, 
                        second_layer_nodes=16, verbose=1, reserve_samples=True):
    """
    Train a neural network model and save it in a comprehensive format with metadata.

    Parameters:
    data_array (np.array): 2D numpy array with the data
    feature_names (list): List of feature column names
    identifiers (np.array): Array of sample identifiers (optional)
    modelname (str): Name for the model and saved files
    base_path (str): Directory path where to save the model
    e_columns_as_inputs (bool): If True, treat E columns as inputs; if False, treat as outputs
    epochs (int): Number of training epochs
    validation_split (float): Fraction of data to use for validation
    batch_size (int): Batch size for training
    first_layer_nodes (int): Number of nodes in first hidden layer
    second_layer_nodes (int): Number of nodes in second hidden layer
    verbose (int): Verbosity level for training
    reserve_samples (bool): Whether to reserve samples for final testing

    Returns:
    dict: Dictionary containing model paths and metadata
    """
    import json
    import time
    from datetime import datetime

    print(f"=== Training and Saving Model: {modelname} ===")

    # Train the model
    model, history, input_features, output_features = build_and_train_neural_network(
        data_array=data_array,
        feature_names=feature_names,
        identifiers=identifiers,
        e_columns_as_inputs=e_columns_as_inputs,
        epochs=epochs,
        validation_split=validation_split,
        verbose=verbose,
        batch_size=batch_size,
        first_layer_nodes=first_layer_nodes,
        second_layer_nodes=second_layer_nodes,
        reserve_samples=reserve_samples
    )

    if base_path is None:
        base_path = os.getcwd()

    # Create MLmodels directory and model directory within it
    mlmodels_dir = os.path.join(base_path, "MLmodels")
    os.makedirs(mlmodels_dir, exist_ok=True)

    model_dir = os.path.join(mlmodels_dir, f"{modelname}_model")
    os.makedirs(model_dir, exist_ok=True)

    # Save in Keras format (primary format for Keras 3 compatibility)
    keras_model_path = os.path.join(model_dir, f"{modelname}.keras")
    model.save(keras_model_path)
    print(f"Model saved in Keras format: {keras_model_path}")

    # Also save in TensorFlow SavedModel format (for inference with TFSMLayer)
    model_path = os.path.join(model_dir, "saved_model")
    try:
        model.export(model_path)
        print(f"Model saved in SavedModel format: {model_path}")
    except Exception as e:
        print(f"Warning: Could not save SavedModel format: {e}")
        model_path = None

    # Save model weights separately
    weights_path = os.path.join(model_dir, f"{modelname}.weights.h5")
    model.save_weights(weights_path)
    print(f"Model weights saved: {weights_path}")

    # Prepare comprehensive metadata
    metadata = {
        'model_info': {
            'name': modelname,
            'created_timestamp': datetime.now().isoformat(),
            'tensorflow_version': keras.__version__,
        },
        'data_info': {
            'total_samples': len(data_array),
            'total_features': len(feature_names),
            'input_features': input_features,
            'output_features': output_features,
            'all_feature_names': feature_names,
            'e_columns_as_inputs': e_columns_as_inputs
        },
        'model_architecture': {
            'input_shape': len(input_features),
            'output_shape': len(output_features),
            'first_layer_nodes': first_layer_nodes,
            'second_layer_nodes': second_layer_nodes,
            'activation_function': 'tanh',
            'output_activation': 'linear',
            'optimizer': 'adam',
            'loss_function': 'mse',
            'metrics': ['mae']
        },
        'training_config': {
            'epochs': epochs,
            'batch_size': batch_size,
            'validation_split': validation_split,
            'early_stopping': True,
            'early_stopping_patience': 10
        },
        'training_results': {
            'epochs_trained': len(history.history['loss']),
            'final_training_loss': float(history.history['loss'][-1]),
            'final_validation_loss': float(history.history['val_loss'][-1]) if 'val_loss' in history.history else None,
            'min_validation_loss': float(min(history.history['val_loss'])) if 'val_loss' in history.history else None,
            'min_val_loss_epoch': int(np.argmin(history.history['val_loss']) + 1) if 'val_loss' in history.history else None
        },
        'file_paths': {
            'saved_model': model_path,
            'keras_model': keras_model_path,
            'weights': weights_path,
            'metadata': os.path.join(model_dir, f"{modelname}_metadata.json"),
            'training_history': os.path.join(model_dir, f"{modelname}_training_history.json"),
            'model_summary': os.path.join(model_dir, f"{modelname}_summary.txt")
        }
    }

    # Save metadata as JSON
    metadata_path = os.path.join(model_dir, f"{modelname}_metadata.json")
    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(metadata_path), exist_ok=True)

        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        # Verify the file was created and is readable
        if os.path.exists(metadata_path) and os.path.getsize(metadata_path) > 0:
            # Test that we can read it back
            with open(metadata_path, 'r') as f:
                test_metadata = json.load(f)
            print(f"Model metadata saved and verified: {metadata_path}")
        else:
            raise Exception("Metadata file was not created properly")

    except Exception as e:
        print(f"Error saving metadata: {e}")
        # Try to save with a different approach
        try:
            metadata_backup_path = os.path.join(model_dir, f"{modelname}_metadata_backup.json")
            with open(metadata_backup_path, 'w') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            print(f"Model metadata saved to backup location: {metadata_backup_path}")
            metadata_path = metadata_backup_path
        except Exception as e2:
            print(f"Error saving metadata backup: {e2}")
            raise

    # Save training history as JSON
    history_path = os.path.join(model_dir, f"{modelname}_training_history.json")
    history_dict = {
        'loss': [float(x) for x in history.history['loss']],
        'mae': [float(x) for x in history.history['mae']],
    }
    if 'val_loss' in history.history:
        history_dict['val_loss'] = [float(x) for x in history.history['val_loss']]
        history_dict['val_mae'] = [float(x) for x in history.history['val_mae']]

    with open(history_path, 'w') as f:
        json.dump(history_dict, f, indent=2)
    print(f"Training history saved: {history_path}")

    # Save model summary as text
    summary_path = os.path.join(model_dir, f"{modelname}_summary.txt")
    with open(summary_path, 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))
    print(f"Model summary saved: {summary_path}")

    # Save training history plot
    plot_path = plot_training_history(
        history, 
        save_path=model_dir, 
        filename=f"{modelname}_training_history.png"
    )
    metadata['file_paths']['training_plot'] = plot_path

    # Update metadata file with plot path
    try:
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        # Verify the updated file
        if os.path.exists(metadata_path) and os.path.getsize(metadata_path) > 0:
            with open(metadata_path, 'r') as f:
                test_metadata = json.load(f)
            print(f"Model metadata updated and verified: {metadata_path}")
        else:
            print(f"Warning: Metadata file update may have failed")

    except Exception as e:
        print(f"Error updating metadata with plot path: {e}")
        # Continue without failing the entire function
        print(f"Model saved successfully despite metadata update issue")

    print(f"\n=== Model Saved Successfully ===")
    print(f"Model directory: {model_dir}")
    print(f"Use tf.keras.models.load_model('{keras_model_path}') to load the model")
    if model_path:
        print(f"For inference-only use: keras.layers.TFSMLayer('{model_path}', call_endpoint='serving_default')")

    return {
        'model_dir': model_dir,
        'model_path': model_path,
        'metadata': metadata,
        'model': model,
        'history': history,
        'input_features': input_features,
        'output_features': output_features
    }


# Main execution block
if __name__ == "__main__":
    print("=== Loading ML Database ===")

    base_path = "/Users/oliverreed/Library/CloudStorage/GoogleDrive-or280@cam.ac.uk/Shared drives/MET - ShapeMemoryAlloyResearch/Useful/Python Scripts/MLFiles"
    modelname = 'Ms_prediction'
    filter_by_column = 'P2'
    selected_columns = ['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10', 'D11', 'D12', 'D13', 'D14', 'D15', 'D16', 'D17', 'D18', 'D19', 'D20', 'D21', 'D22', 'D23', 'D24', 'D25', 'D26', 'D27', 'D28', 'D29', 'D30', 'D31', 'D32', 'D33', 'P2']

    # Select specific columns with row filtering
    print("\nLoading specific features (composition + one property) with P2 filtering:")
    try:
        csv_file_path = os.path.join(base_path, "ML_Database.csv")
        identifiers_sel, features_sel, data_sel, headers_sel = load_ml_data(
            csv_file_path=csv_file_path,
            filter_by_column=filter_by_column,
            selected_columns=selected_columns,
            fill_nan_with_median=True
        )
        
        print(f"Data preview:")
        for i, sample_id in enumerate(identifiers_sel[:3]):  # Show first 3 samples
            print(f"  {sample_id}: {data_sel[i]}")

        # Tune hyperparameters
        '''
        try:
            results = tune_hyperparameters(
                data_sel, features_sel, 
                identifiers=identifiers_sel,
                e_columns_as_inputs=True,
                epochs=1500,
                validation_split=0.2,
                verbose=0,
                save_path=base_path,
                modelname=modelname
            )
        '''

        # Train and save final model
        print("\n=== Training and Saving Final Model ===")
        try:
            model_info = train_and_save_model(
                data_array=data_sel,
                feature_names=features_sel,
                identifiers=identifiers_sel,
                modelname=modelname,
                base_path=base_path,
                e_columns_as_inputs=True,
                epochs=1500,
                validation_split=0.2,
                batch_size=8,
                first_layer_nodes=64,
                second_layer_nodes=32,
                verbose=0,
                reserve_samples=False
            )
            print("\n=== Model Training and Saving Complete ===")
        except Exception as e:
            print(f"Error training and saving model: {e}")

    except Exception as e:
        print(f"Error loading selected data: {e}")

    print("\n=== Database Loading Complete ===")