import numpy as np
import torch
import torch.nn as nn
import joblib
import os
import math
from utils import DataPoint # Provided by the competition environment

# --- Configuration ---
# !!! IMPORTANT: These values MUST match train.py
N_FEATURES = 32  # This must match the N_FEATURES from your data

# --- Model Hyperparameters (Must match train.py) ---
HIDDEN_SIZE = 512         # Increased from 256 → 512 for more capacity
NUM_LAYERS = 3
DROPOUT_RATE = 0.15       # Reduced from 0.30 → 0.15 to preserve more info flow
USE_DELTA_FEATURES = True # If True, input_size doubles to N_FEATURES * 2
MODEL_TYPE = "GRU"        # Options: "GRU" or "LSTM"

# Effective input size depends on whether delta features are used
INPUT_SIZE = N_FEATURES * 2 if USE_DELTA_FEATURES else N_FEATURES


# =========================
# GRU Model Definition
# =========================
class GRUModel(nn.Module):
    """
    Original GRU model with added improvements:
    - LayerNorm for training stability
    - Residual/skip connection to learn deltas instead of absolute values
    """
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_size = input_size
        self.output_size = output_size

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, output_size)

    @torch.no_grad()
    def init_hidden(self, batch_size, device):
        dtype = next(self.parameters()).dtype
        return torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device, dtype=dtype)

    def forward(self, x, hidden):
        """
        Forward pass with LayerNorm and residual connection.
        x shape: (B=1, T=1, F_in)
        hidden shape: (L, B=1, H)
        """
        # out shape: (1, 1, H)
        out, new_hidden = self.gru(x, hidden)
        
        # Take just the last time step's output
        # out shape: (1, H)
        out = out[:, -1, :] 
        
        # Apply Layer Normalization for stability
        out = self.layer_norm(out)
        
        # pred shape: (1, F_out)
        pred = self.linear(out)
        
        # Residual/skip connection: predict delta from current input
        # Add the raw features (first N_FEATURES of input) to learn changes
        pred = pred + x[:, -1, :self.output_size]
        
        return pred, new_hidden


# =========================
# LSTM Model Definition
# =========================
class LSTMModel(nn.Module):
    """
    LSTM model — has a separate cell state + hidden state (3 gates vs GRU's 2).
    Better at capturing long-term dependencies for subtle patterns.
    Also includes LayerNorm and residual connection.
    """
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_size = input_size
        self.output_size = output_size

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, output_size)

    @torch.no_grad()
    def init_hidden(self, batch_size, device):
        """
        LSTM needs TWO states: hidden state (h) and cell state (c).
        Returns a tuple (h_0, c_0).
        """
        dtype = next(self.parameters()).dtype
        h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device, dtype=dtype)
        c_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device, dtype=dtype)
        return (h_0, c_0)

    def forward(self, x, hidden):
        """
        Forward pass with LayerNorm and residual connection.
        x shape: (B=1, T=1, F_in)
        hidden: tuple of (h, c), each shape (L, B=1, H)
        """
        # out shape: (1, 1, H), new_hidden is tuple (h_n, c_n)
        out, new_hidden = self.lstm(x, hidden)
        
        # Take just the last time step's output
        # out shape: (1, H)
        out = out[:, -1, :]
        
        # Apply Layer Normalization for stability
        out = self.layer_norm(out)
        
        # pred shape: (1, F_out)
        pred = self.linear(out)
        
        # Residual/skip connection: predict delta from current input
        pred = pred + x[:, -1, :self.output_size]
        
        return pred, new_hidden


# --- Main Prediction Class ---
class PredictionModel:
    """
    The main class required by the competition environment.
    Supports both GRU and LSTM models via MODEL_TYPE config.
    """
    def __init__(self):
        """
        Initialize the model, load weights, and set up thread limits.
        """
        # Enforce single-core CPU operation as required
        torch.set_num_threads(1)
        self.device = torch.device('cpu')
        
        # Select model architecture based on MODEL_TYPE
        if MODEL_TYPE == "LSTM":
            self.model = LSTMModel(
                input_size=INPUT_SIZE,
                hidden_size=HIDDEN_SIZE,
                num_layers=NUM_LAYERS,
                output_size=N_FEATURES,
                dropout=DROPOUT_RATE
            ).to(self.device)
            model_path = 'lstm_model.pth'
        else:  # Default: GRU
            self.model = GRUModel(
                input_size=INPUT_SIZE,
                hidden_size=HIDDEN_SIZE,
                num_layers=NUM_LAYERS,
                output_size=N_FEATURES,
                dropout=DROPOUT_RATE
            ).to(self.device)
            model_path = 'gru_model.pth'
        
        # Load the trained model weights
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        # Load the fitted scaler
        scaler_path = 'scaler.joblib'
        self.scaler = joblib.load(scaler_path)
        
        # Dictionary to store hidden states for each sequence
        self.hidden_states = {}
        
        # Dictionary to store previous scaled states (for delta features)
        self.prev_scaled_states = {}

    def predict(self, data_point: DataPoint) -> np.ndarray | None:
        """
        Generate a prediction for a single timestep.
        """
        
        # --- 1. Get Sequence ID and Step ---
        seq_ix = data_point.seq_ix
        step_in_seq = int(data_point.step_in_seq)

        # --- 2. Manage Hidden State ---
        if step_in_seq == 0 or seq_ix not in self.hidden_states:
            # Clean up old hidden states to free memory
            old_keys = [k for k in self.hidden_states if k != seq_ix]
            for k in old_keys:
                del self.hidden_states[k]
            if seq_ix in self.prev_scaled_states:
                # Also clean old delta states
                old_delta_keys = [k for k in self.prev_scaled_states if k != seq_ix]
                for k in old_delta_keys:
                    del self.prev_scaled_states[k]
            
            # Initialize fresh hidden state for new sequence
            self.hidden_states[seq_ix] = self.model.init_hidden(batch_size=1, device=self.device)
        
        current_hidden = self.hidden_states[seq_ix]

        # --- 3. Preprocess Input State ---
        state_reshaped = data_point.state.reshape(1, -1)
        scaled_state = self.scaler.transform(state_reshaped)
        
        # --- 4. Build Input (with optional delta features) ---
        if USE_DELTA_FEATURES:
            if seq_ix in self.prev_scaled_states:
                delta = scaled_state - self.prev_scaled_states[seq_ix]
            else:
                delta = np.zeros_like(scaled_state)
            # Concatenate raw state + delta → shape (1, 64)
            model_input = np.concatenate([scaled_state, delta], axis=1)
            self.prev_scaled_states[seq_ix] = scaled_state.copy()
        else:
            model_input = scaled_state
        
        # Convert to tensor shape [1, 1, INPUT_SIZE]
        x_tensor = torch.tensor(model_input, dtype=torch.float32).view(1, 1, -1).to(self.device)

        # --- 5. Model Inference ---
        with torch.no_grad():
            out_tensor, new_hidden = self.model(x_tensor, current_hidden)
            prediction_scaled = out_tensor.cpu().numpy()
        
        # --- 6. Update and Store State ---
        if MODEL_TYPE == "LSTM":
            # LSTM hidden is a tuple (h, c) — detach both
            self.hidden_states[seq_ix] = (new_hidden[0].detach(), new_hidden[1].detach())
        else:
            self.hidden_states[seq_ix] = new_hidden.detach()

        # --- 7. Handle Warmup vs. Prediction ---
        if not data_point.need_prediction:
            return None

        # --- 8. Post-process and Return Prediction ---
        prediction_original_scale = self.scaler.inverse_transform(prediction_scaled)
        return prediction_original_scale.flatten()