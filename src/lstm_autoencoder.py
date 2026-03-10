import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, n_features: int, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_features = n_features

        # The LSTM Layer
        # batch_first=True means we expect our 3D tensor to have the Batch
        # size as the 0th dimension
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )

    def forward(self, x):
        # x shape: (Batch, Sequence_Length, Features) -> e.g. (32, 100, 38)

        # Pass the 3D tensor through the LSTM
        # 'lstm_out' contains the hidden states for ALL 100 timesteps.
        # 'hidden' contains the final state at the very last timestep.
        lstm_out, (hidden, cell) = self.lstm(x)

        # We only care about the very last hidden state, because it has "read"
        # the entire sequence and compressed it into a single vector.
        # hidden shape: (1, Batch, Hidden_Dim). We reshape it to (Batch, Hidden_Dim).
        return hidden.squeeze(0)


class Decoder(nn.Module):
    def __init__(self, seq_len: int, hidden_dim: int, n_features: int):
        super().__init__()
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )

        # The output of the LSTM is still in the 'hidden_dim' space.
        # We use a standard linear layer (matrix multiplication) to project it back to 38 sensors
        self.output_layer = nn.Linear(hidden_dim, n_features)

    def forward(self, x):
        # x shape: (Batch, Hidden_Dim) -> e.g., (32, 16)

        # We need to unroll this single vector back into a sequence of 100 steps.
        # So, we literally duplicate it 100 times.
        # New shape: (Batch, Sequence_Length, Hidden_Dim) -> (32, 100, 16)
        x = x.unsqueeze(1).repeat(1, self.seq_len, 1)

        # Pass the repeated vectors through the Decoder LSTM
        lstm_out, (hidden, cell) = self.lstm(x)

        # lstm_out shape: (Batch, Sequence_Length, Hidden_Dim)
        # We push every single timestep through the linear layer to get back to our 38 features.
        # Final shape: (Batch, Sequence_Length, Features) -> (32, 100, 38)
        return self.output_layer(lstm_out)


class LSTMAutoencoder(nn.Module):
    def __init__(self, seq_len: int, n_features: int, hidden_dim: int):
        super().__init__()
        self.encoder = Encoder(n_features, hidden_dim)
        self.decoder = Decoder(seq_len, hidden_dim, n_features)

    def forward(self, x):
        # The forward pass simply connects the pipes
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def train_lstm_autoencoder(model: LSTMAutoencoder,
                           train_loader: torch.utils.data.DataLoader,
                           num_epochs: int = 20,
                           learning_rate: float = 1e-3):

    # 1. The Loss Function: Mean Squared Error (MSE)
    # This calculates how badly the model failed to reconstruct the input.
    criterion = nn.MSELoss()

    # 2. The Optimizer: Adam
    # Adam is the mechanice. It takes the error from the Loss Function and figures
    # out exactly how much the turn the wrenches (update the weights) to reduce that error.
    # the "learning_rate" is how aggressively the mechanic turns the wrenches.
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # An "epoch" is one full pass through the entire training dataset
    for epoch in range(num_epochs):
        # We process the data in small batches
        for batch_data in train_loader:
            batch = batch_data[0]

            # Step A: Clear the mechanic's tools from the last batch
            optimizer.zero_grad()

            # Step B: The Forward Pass (Compress and Reconstruct)
            # Notice we pass 'batch' into the model, and it outputs the reconstruction
            reconstruction = model(batch)

            # Step C: Calculate the Error
            # We compare the reconstruction directly against the original batch.
            loss = criterion(reconstruction, batch)

            # Step D: The Backward Pass (Calculus Magic)
            # This calculates the partial derivatives (gradients) for every weight.
            loss.backward()

            # Step E: Update the weights
            # The mechanic actually turns the wrenches based on the calculus.
            optimizer.step()

        # print progress at the end of each epoch
        print(
            f"Epoch [{epoch+1}/{num_epochs}], Reconstruction Error (Loss): {loss.item():.4f}")

    print("Training complete. The weights are locked.")
    return model
