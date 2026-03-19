import numpy as np
import torch
import torch.nn as nn
import joblib
import os
import math
from utils import DataPoint # Provided by the competition environment

# --- Configuration ---
# !!! IMPORTANT: These values MUST match train.py
N_FEATURES = 32 # This must match the N_FEATURES from your data

# --- Model Hyperparameters (Must match train.py) ---
HIDDEN_SIZE = 256
NUM_LAYERS = 3
DROPOUT_RATE = 0.30

# =========================
# Model Definition (Must match train.py)
# =========================
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.linear = nn.Linear(hidden_size, output_size)

    @torch.no_grad()
    def init_hidden(self, batch_size, device):
        dtype = next(self.parameters()).dtype
        return torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device, dtype=dtype)

    def forward(self, x, hidden):
        """
        Modified forward pass for step-by-step inference.
        x shape: (B=1, T=1, F_in)
        hidden shape: (L, B=1, H)
        """
        # out shape: (1, 1, H)
        out, new_hidden = self.gru(x, hidden)
        
        # Take just the last time step's output
        # out shape: (1, H)
        out = out[:, -1, :] 
        
        # pred shape: (1, F_out)
        pred = self.linear(out)
        return pred, new_hidden

# --- Main Prediction Class ---
class PredictionModel:
    """
    The main class required by the competition environment.
    """
    def __init__(self):
        """
        Initialize the model, load weights, and set up thread limits.
        """
        # Enforce single-core CPU operation as required
        torch.set_num_threads(1)
        self.device = torch.device('cpu')
        
        # Define model architecture
        self.model = GRUModel(
            input_size=N_FEATURES,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            output_size=N_FEATURES,
            dropout=DROPOUT_RATE
        ).to(self.device)
        
        # Load the trained model weights
        # This file name MUST match MODEL_SAVE_PATH in train.py
        model_path = 'gru_model.pth'
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        # Load the fitted scaler
        # This file name MUST match SCALER_SAVE_PATH in train.py
        scaler_path = 'scaler.joblib'
        self.scaler = joblib.load(scaler_path)
        
        # Dictionary to store GRU hidden states for each sequence
        self.hidden_states = {}

    def predict(self, data_point: DataPoint) -> np.ndarray | None:
        """
        Generate a prediction for a single timestep.
        """
        
        # --- 1. Get Sequence ID and Step ---
        seq_ix = data_point.seq_ix
        
        # Cast to int() to prevent slice error if step_in_seq is a float (e.g., 0.0)
        step_in_seq = int(data_point.step_in_seq)

        # --- 2. Manage GRU Hidden State ---
        if step_in_seq == 0 or seq_ix not in self.hidden_states:
            # Initialize hidden state for a new sequence
            self.hidden_states[seq_ix] = self.model.init_hidden(batch_size=1, device=self.device)
        
        current_hidden = self.hidden_states[seq_ix]

        # --- 3. Preprocess Input State ---
        state_reshaped = data_point.state.reshape(1, -1)
        scaled_state = self.scaler.transform(state_reshaped)
        
        # Convert to tensor shape [1, 1, N] (batch=1, seq_len=1, features=N)
        x_tensor = torch.tensor(scaled_state, dtype=torch.float32).view(1, 1, -1).to(self.device)

        # --- 4. Model Inference ---
        prediction_scaled = None
        new_hidden = None
        
        with torch.no_grad():
            # Get model output and new hidden state
            # out_tensor shape is (1, F) from our modified forward pass
            out_tensor, new_hidden = self.model(x_tensor, current_hidden)
            
            prediction_scaled = out_tensor.cpu().numpy()
        
        # --- 5. Update and Store State ---
        # Store the GRU's hidden state for the next step
        self.hidden_states[seq_ix] = new_hidden.detach()

        # --- 6. Handle Warmup vs. Prediction ---
        if not data_point.need_prediction:
            return None # Per competition rules

        # --- 7. Post-process and Return Prediction ---
        # prediction_scaled is (1, F), inverse_transform expects (n, F)
        prediction_original_scale = self.scaler.inverse_transform(prediction_scaled)
        
        # Flatten to 1D NumPy array (F,) as required
        return prediction_original_scale.flatten()