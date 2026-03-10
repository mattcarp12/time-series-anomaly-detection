import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path


class SMDPreprocessor:
    def __init__(self, window_size: int = 100, step_size: int = 1):
        self.window_size = window_size
        self.step_size = step_size
        self.scaler = MinMaxScaler()

    def load_data(self, file_path: str) -> np.ndarray:
        """Loads the raw SMD text file into a numpy array."""
        # SMD features are typically space or comma separated
        df = pd.read_csv(file_path, sep=",", header=None)
        return df.values

    def create_sliding_windows(self, data: np.ndarray) -> np.ndarray:
        """
        Slides a window over the time series to create overlapping sequences.
        Returns shape: (num_samples, window_size, num_features)
        """
        windows = []
        # Slide across the sequence
        for i in range(0, len(data) - self.window_size, self.step_size):
            window = data[i: i + self.window_size]
            windows.append(window)

        return np.array(windows)

    def process_machine(self, train_path: str, test_path: str):
        """Full pipeline to process a single server's data."""
        print(f"Processing: {Path(train_path).name}...")

        # 1. Load Data
        raw_train = self.load_data(train_path)
        raw_test = self.load_data(test_path)

        # 2. Scale Data (Fit ONLY on training data to prevent leakage)
        scaled_train = self.scaler.fit_transform(raw_train)
        scaled_test = self.scaler.transform(raw_test)

        # 3. Create Windows
        X_train = self.create_sliding_windows(scaled_train)
        X_test = self.create_sliding_windows(scaled_test)

        print(f"Train shape: {X_train.shape} | Test shape: {X_test.shape}")
        return X_train, X_test

def read_and_process_smd(train_file: str, test_file: str):
    # Example Usage:
    # Assuming you downloaded the NetManAIOps/OmniAnomaly SMD dataset
    # to a local 'data/' directory.

    TRAIN_FILE = "data/train/machine-1-1.txt"
    TEST_FILE = "data/test/machine-1-1.txt"

    preprocessor = SMDPreprocessor(window_size=100, step_size=1)

    try:
        X_train, X_test = preprocessor.process_machine(TRAIN_FILE, TEST_FILE)

        # Save the processed numpy arrays for the next stage (AWS upload)
        np.save("data/processed/machine-1-1_train.npy", X_train)
        np.save("data/processed/machine-1-1_test.npy", X_test)
        print("Preprocessing complete and saved successfully.")

    except FileNotFoundError:
        print("Data files not found. Ensure you have downloaded the SMD dataset.")

def read_and_process_dummy():
    # Generate fake server data: 1000 minutes, 38 sensors
    print("Generating fake training and testing data...")
    fake_train = np.random.rand(1000, 38)
    fake_test = np.random.rand(500, 38)
    
    # Save them temporarily as CSVs to mimic the real files
    pd.DataFrame(fake_train).to_csv("fake_train.csv", index=False, header=False)
    pd.DataFrame(fake_test).to_csv("fake_test.csv", index=False, header=False)
    
    # Initialize our class with a window of 100 and a step of 1
    preprocessor = SMDPreprocessor(window_size=100, step_size=1)
    
    # Run the pipeline
    X_train, X_test = preprocessor.process_machine("fake_train.csv", "fake_test.csv")
    
    # Clean up the fake files
    Path("fake_train.csv").unlink()
    Path("fake_test.csv").unlink()



if __name__ == "__main__":
    # Uncomment the line below to run with real SMD data (ensure you have it downloaded)
    # read_and_process_smd("data/train/machine-1-1.txt", "data/test/machine-1-1.txt")
    
    # For testing purposes, we can run the dummy data generator
    read_and_process_dummy()    
