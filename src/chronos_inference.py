import torch
import numpy as np
import pandas as pd
from chronos import ChronosPipeline
from sklearn.metrics import precision_score, recall_score, f1_score
from preprocess_smd import SMDPreprocessor


def run_chronos_baseline():
    print("--- STEP 1: INITIALIZING CHRONOS ---")
    # Load the Amazon Chronos-Bolt model.
    # 'amazon/chronos-bolt-small' is lightweight enough for CPU inference.
    pipeline = ChronosPipeline.from_pretrained(
        "amazon/chronos-t5-small",
        device_map="cpu",  # Using CPU since we are in a devcontainer
        dtype=torch.float32,
    )

    print("\n--- STEP 2: PREPARING THE DATA ---")
    WINDOW_SIZE = 100
    # Let's test on just the first 1000 windows to save compute time locally
    TEST_LIMIT = 1000

    TRAIN_FILE = "data/smd/ServerMachineDataset/train/machine-1-1.txt"
    TEST_FILE = "data/smd/ServerMachineDataset/test/machine-1-1.txt"
    LABEL_FILE = "data/smd/ServerMachineDataset/test_label/machine-1-1.txt"

    # We use our preprocessor just to load and scale the data cleanly
    preprocessor = SMDPreprocessor(window_size=WINDOW_SIZE, step_size=1)
    _, X_test_np = preprocessor.process_machine(TRAIN_FILE, TEST_FILE)

    # Slice the data to our TEST_LIMIT
    X_test_np = X_test_np[:TEST_LIMIT]

    # Load and align the answer key just like before
    raw_labels = pd.read_csv(LABEL_FILE, header=None).dropna().values.flatten()
    start_idx = WINDOW_SIZE - 1
    aligned_labels = raw_labels[start_idx: start_idx + TEST_LIMIT]

    print(f"Testing on {TEST_LIMIT} windows...")

    print("\n--- STEP 3: ZERO-SHOT INFERENCE ---")
    # We will store a boolean True/False for each window
    anomalies = np.zeros(TEST_LIMIT, dtype=bool)

    # Chronos is univariate, so we loop through our windows
    for i in range(TEST_LIMIT):
        if i % 100 == 0:
            print(f"Processing window {i}/{TEST_LIMIT}...")

        # Grab the current window: shape (100 timesteps, 38 features)
        window = X_test_np[i]

        # The first 99 steps are the context. The 100th step is the "actual" value we want to verify.
        context = window[:99, :]  # shape: (99, 38)
        actual_next_step = window[99, :]  # shape: (38,)

        # Chronos expects input shape (batch_size, sequence_length).
        # We transpose our context so each of the 38 sensors becomes its own batch item!
        # Context shape becomes: (38 sensors, 99 timesteps)
        context_tensor = torch.tensor(context.T, dtype=torch.float32).contiguous()

        # Forecast 1 step into the future for all 38 sensors.
        # We ask Chronos to simulate 100 possible futures to build a strong probability curve.
        with torch.no_grad():
            forecast = pipeline.predict(
                context_tensor, prediction_length=1, num_samples=20)

        # forecast shape: (38 sensors, 100 samples, 1 step)
        # We calculate the 1st and 99th percentile across the 100 samples (dim=1)
        low_bound_tensor = torch.quantile(
            forecast, 0.01, dim=1)  # shape: (38, 1)
        high_bound_tensor = torch.quantile(
            forecast, 0.99, dim=1)  # shape: (38, 1)

        # Squeeze out that useless trailing 1-dimension and convert to NumPy
        low_bound = low_bound_tensor.squeeze(-1).numpy()   # shape: (38,)
        high_bound = high_bound_tensor.squeeze(-1).numpy()  # shape: (38,)

        # Now BOTH actual_next_step and low_bound are shape (38,)!
        is_anomaly = (actual_next_step < low_bound) | (
            actual_next_step > high_bound)

        # If any single sensor breached its predicted bounds, flag the minute
        if np.any(is_anomaly):
            anomalies[i] = True

    print("\n--- STEP 4: GRADING CHRONOS ---")
    predictions = anomalies.astype(int)

    precision = precision_score(aligned_labels, predictions, zero_division=0)
    recall = recall_score(aligned_labels, predictions, zero_division=0)
    f1 = f1_score(aligned_labels, predictions, zero_division=0)

    print(f"Chronos Precision: {precision:.4f}")
    print(f"Chronos Recall:    {recall:.4f}")
    print(f"Chronos F1-Score:  {f1:.4f}")


if __name__ == "__main__":
    run_chronos_baseline()
