import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
import pandas as pd

# Import the pieces we just built
from preprocess_smd import SMDPreprocessor
from lstm_autoencoder import LSTMAutoencoder, train_lstm_autoencoder


def get_anomaly_scores(model, data_tensor):
    """Runs data through the model and returns the anomaly score for each window."""
    # 1. Turn on Evaluation Mode
    model.eval()

    # 2. Turn off gradient tracking to save memory
    with torch.no_grad():
        # 3. Get the reconstructions
        reconstructions = model(data_tensor)

        # 4. Calculate the Mean Absolute Error (MAE)
        # We subtract the Input from the Output.
        # dim=(1, 2) means we average the error across all 100 timesteps and 38 sensors.
        # This boils the massive 3D tensor down to exactly ONE error score per window.
        error = torch.mean(
            torch.abs(reconstructions - data_tensor), dim=(1, 2))

    # Convert back to a standard NumPy array for easy math
    return error.numpy()


def run_pipeline():
    # 1. Define the parameters
    WINDOW_SIZE = 100
    STEP_SIZE = 1
    BATCH_SIZE = 32
    HIDDEN_DIM = 16   # The size of our bottleneck
    NUM_FEATURES = 38  # SMD has 38 sensors

    # We will use just one machine's data from the SMD dataset for now
    TRAIN_FILE = "data/smd/ServerMachineDataset/train/machine-1-1.txt"
    TEST_FILE = "data/smd/ServerMachineDataset/test/machine-1-1.txt"

    # 2. Preprocess the Data
    print("--- STEP 1: PREPROCESSING ---")
    preprocessor = SMDPreprocessor(
        window_size=WINDOW_SIZE, step_size=STEP_SIZE)
    X_train_np, X_test_np = preprocessor.process_machine(TRAIN_FILE, TEST_FILE)

    # 3. Convert to PyTorch Tensors
    # We use torch.FloatTensor because neural networks expect 32-bit floating point math
    print("\n--- STEP 2: CONVERTING TO TENSORS ---")
    X_train_tensor = torch.FloatTensor(X_train_np)
    # We will save the test tensor for later when we actually hunt for anomalies
    X_test_tensor = torch.FloatTensor(X_test_np)

    # 4. Create the DataLoader
    # TensorDataset wraps our tensor. DataLoader handles the batching and shuffling.
    # We shuffle the training data so the model doesn't just memorize the chronological order!
    train_dataset = TensorDataset(X_train_tensor)

    # Note: TensorDataset returns a tuple, so we unpack it in the loop later.
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 5. Initialize the Model
    print("\n--- STEP 3: BUILDING THE ENGINE ---")
    model = LSTMAutoencoder(seq_len=WINDOW_SIZE,
                            n_features=NUM_FEATURES, hidden_dim=HIDDEN_DIM)

    # 6. Train the Model!
    print("\n--- STEP 4: TRAINING ---")
    # We'll run it for 5 epochs just to watch it learn quickly
    trained_model = train_lstm_autoencoder(model, train_loader, num_epochs=10)

    print("\n--- STEP 5: CALCULATING THE THRESHOLD ---")
    # Pass the normal training data back through to see what "normal" errors look like
    train_scores = get_anomaly_scores(trained_model, X_train_tensor)

    # Set the tripwire at the 99th percentile of normal data
    threshold = np.percentile(train_scores, 99.9)
    print(f"Calculated Anomaly Threshold: {threshold:.4f}")

    print("\n--- STEP 6: HUNTING FOR ANOMALIES IN TEST DATA ---")
    # Pass the unseen test data through the engine
    test_scores = get_anomaly_scores(trained_model, X_test_tensor)

    # Create a boolean array (True/False) where the score exceeds the threshold
    anomalies = test_scores > threshold
    num_anomalies = np.sum(anomalies)

    print(f"Scanned {len(test_scores)} test windows.")
    print(f"🚨 Detected {num_anomalies} anomalous events! 🚨")

    print("\n--- STEP 7: GRADING THE MODEL (THE ANSWER KEY) ---")

    # 1. Load the actual ground-truth labels
    LABEL_FILE = "data/smd/ServerMachineDataset/test_label/machine-1-1.txt"
    # Read the file and flatten it into a simple 1D NumPy array
    raw_labels = pd.read_csv(LABEL_FILE, header=None).dropna().values.flatten()

    # 2. Align the labels with our sliding windows
    # We start at the end of the first window (WINDOW_SIZE - 1)
    start_idx = WINDOW_SIZE - 1

    # We strictly slice the array to exactly match the length of our anomalies
    aligned_labels = raw_labels[start_idx: start_idx + len(anomalies)]

    print(
        f"Alignment Check: {len(anomalies)} predictions vs {len(aligned_labels)} labels.")

    # 3. Calculate the Golden Metrics
    # 'anomalies' is our boolean array (True/False). Convert to 1s and 0s.
    predictions = anomalies.astype(int)

    precision = precision_score(aligned_labels, predictions)
    recall = recall_score(aligned_labels, predictions)
    f1 = f1_score(aligned_labels, predictions)

    print(f"Precision: {precision:.4f} (When it alerts, is it right?)")
    print(f"Recall:    {recall:.4f} (Did it miss any real crashes?)")
    print(f"F1-Score:  {f1:.4f} (The ultimate harmonic balance)")


if __name__ == "__main__":
    run_pipeline()
