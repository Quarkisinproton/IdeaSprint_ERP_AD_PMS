"""
ERR-CC v4.0 — Predictive Maintenance Models
=============================================
Contains:
  - LSTMRULModel (PyTorch): LSTM-based Remaining Useful Life predictor
  - prepare_lstm_sequences: Sliding window data preparation  
  - train_lstm: Training loop with validation
  - monte_carlo_rul_simulation: Probabilistic RUL with confidence bands
  - fit_prophet_engine: Prophet / fallback trend forecasting per engine
"""

import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler

# --- PyTorch availability ---
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("  [WARNING] PyTorch not available. LSTM RUL model will use XGBoost fallback.")

# --- Prophet availability ---
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    print("  [WARNING] Prophet not available. Using numpy linear trend fallback.")


# ============================================================
# 1. LSTM RUL MODEL (PyTorch)
# ============================================================

if TORCH_AVAILABLE:
    class LSTMRULModel(nn.Module):
        """
        LSTM-based Remaining Useful Life predictor.
        Architecture: LSTM(input, hidden=64, layers=2, dropout=0.2) → Linear(64, 1)
        """
        def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
            super(LSTMRULModel, self).__init__()
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout
            )
            self.fc = nn.Linear(hidden_size, 1)
        
        def forward(self, x):
            # x shape: (batch, seq_len, features)
            lstm_out, (h_n, c_n) = self.lstm(x)
            # Use last hidden state from the final layer
            last_hidden = h_n[-1]  # shape: (batch, hidden_size)
            out = self.fc(last_hidden)
            return out.squeeze(-1)


def prepare_lstm_sequences(df_nasa, sensor_cols, window=30):
    """
    Create sliding window sequences for LSTM training.
    
    For each engine, creates sequences X[i] = sensors[i:i+window], y[i] = RUL[i+window]
    
    Args:
        df_nasa: DataFrame with engine telemetry
        sensor_cols: list of sensor column names
        window: sequence length (number of cycles)
    
    Returns:
        X: numpy array of shape (n_samples, window, n_features)
        y: numpy array of shape (n_samples,)
        scaler: fitted StandardScaler for sensor inputs
    """
    scaler = StandardScaler()
    
    all_X, all_y = [], []
    
    # Fit scaler on all sensor data first
    all_sensor_data = df_nasa[sensor_cols].values
    scaler.fit(all_sensor_data)
    
    for eid in df_nasa['EngineID'].unique():
        engine_data = df_nasa[df_nasa['EngineID'] == eid].sort_values('Cycle')
        sensor_vals = scaler.transform(engine_data[sensor_cols].values)
        rul_vals = engine_data['RUL'].values
        
        # Create sliding windows
        for i in range(len(sensor_vals) - window):
            seq = sensor_vals[i:i + window]
            target = rul_vals[i + window]
            all_X.append(seq)
            all_y.append(target)
    
    X = np.array(all_X, dtype=np.float32)
    y = np.array(all_y, dtype=np.float32)
    
    return X, y, scaler


def train_lstm(X, y, epochs=30, lr=1e-3, batch_size=64):
    """
    Train the LSTM RUL model.
    
    Args:
        X: numpy array (n_samples, window, n_features)
        y: numpy array (n_samples,)
        epochs: training epochs (default 30, reduced for speed without accuracy loss)
        lr: learning rate
        batch_size: mini-batch size
    
    Returns:
        model: trained LSTMRULModel
        train_rmse: final training RMSE
    """
    if not TORCH_AVAILABLE:
        print("    [FALLBACK] PyTorch unavailable.")
        return None, 0.0
    
    torch.manual_seed(42)
    np.random.seed(42)
    
    n_features = X.shape[2]
    model = LSTMRULModel(input_size=n_features)
    
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.FloatTensor(y)
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        n_batches = 0
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            output = model(batch_X)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        
        avg_loss = epoch_loss / max(n_batches, 1)
        if (epoch + 1) % 10 == 0:
            rmse = np.sqrt(avg_loss)
            print(f"      Epoch {epoch+1}/{epochs} — RMSE: {rmse:.2f}")
    
    # Compute final RMSE
    model.eval()
    with torch.no_grad():
        preds = model(X_tensor).numpy()
    train_rmse = np.sqrt(np.mean((y - preds) ** 2))
    
    return model, train_rmse


def predict_lstm(model, engine_data, sensor_cols, scaler, window=30):
    """
    Get LSTM RUL predictions for a single engine's data.
    
    Returns:
        predictions: array of predicted RUL values (one per valid window position)
        valid_cycles: array of cycle numbers corresponding to predictions
    """
    if not TORCH_AVAILABLE or model is None:
        return np.array([]), np.array([])
    
    sensor_vals = scaler.transform(engine_data[sensor_cols].values)
    cycles = engine_data['Cycle'].values
    
    if len(sensor_vals) < window:
        return np.array([]), np.array([])
    
    sequences = []
    valid_cycles = []
    for i in range(len(sensor_vals) - window):
        sequences.append(sensor_vals[i:i + window])
        valid_cycles.append(cycles[i + window])
    
    X = np.array(sequences, dtype=np.float32)
    X_tensor = torch.FloatTensor(X)
    
    model.eval()
    with torch.no_grad():
        preds = model(X_tensor).numpy()
    
    return preds, np.array(valid_cycles)


# ============================================================
# 2. MONTE CARLO RUL SIMULATION
# ============================================================

def monte_carlo_rul_simulation(engine_df, xgb_model, sensor_cols, rul_features, 
                                training_residual_std=None, n_sims=500):
    """
    Monte Carlo simulation for probabilistic RUL estimation.
    
    For each engine, perturb the last known sensor readings with noise
    sampled from training residual distribution, then predict RUL.
    
    Args:
        engine_df: DataFrame of a single engine's data
        xgb_model: trained XGBoost RUL regressor
        sensor_cols: list of sensor column names
        rul_features: list of feature names for the XGBoost model
        training_residual_std: dict of {feature: std} from training residuals
        n_sims: number of Monte Carlo simulations
    
    Returns:
        dict with keys: 'p10', 'p50', 'p90', 'mean', 'std', 
        'failure_prob_30': probability of RUL < 30
    """
    last_row = engine_df.iloc[-1]
    
    # Default noise if no residual info available
    if training_residual_std is None:
        training_residual_std = {col: 0.05 * abs(last_row.get(col, 1)) 
                                  for col in sensor_cols}
    
    rul_predictions = []
    for _ in range(n_sims):
        # Perturb sensor readings with Gaussian noise
        perturbed = {}
        for f in rul_features:
            base_val = last_row.get(f, 0)
            noise_std = training_residual_std.get(f, 0.01 * abs(base_val) + 0.1)
            perturbed[f] = base_val + np.random.normal(0, noise_std)
        
        X_sim = pd.DataFrame([perturbed])[rul_features]
        pred_rul = max(0, xgb_model.predict(X_sim)[0])
        rul_predictions.append(pred_rul)
    
    rul_predictions = np.array(rul_predictions)
    
    return {
        'p10': float(np.percentile(rul_predictions, 10)),
        'p50': float(np.percentile(rul_predictions, 50)),
        'p90': float(np.percentile(rul_predictions, 90)),
        'mean': float(np.mean(rul_predictions)),
        'std': float(np.std(rul_predictions)),
        'failure_prob_30': float(np.mean(rul_predictions < 30))
    }


# ============================================================
# 3. PROPHET / FALLBACK TREND FORECASTING
# ============================================================

def fit_prophet_engine(engine_df, sensor_col='Sensor_3_HPC', forecast_periods=30):
    """
    Fit Prophet on a single engine's sensor data to forecast future degradation.
    Falls back to numpy linear regression if Prophet is unavailable.
    
    Args:
        engine_df: DataFrame for one engine, sorted by Cycle
        sensor_col: sensor column to forecast
        forecast_periods: number of cycles to forecast ahead
    
    Returns:
        dict with keys: 'forecast_df', 'trend_slope', 'accelerating_degradation'
    """
    cycles = engine_df['Cycle'].values
    values = engine_df[sensor_col].values
    
    if PROPHET_AVAILABLE:
        try:
            # Prophet requires 'ds' and 'y' columns
            base_date = pd.Timestamp('2025-01-01')
            ds = [base_date + pd.Timedelta(days=int(c)) for c in cycles]
            
            prophet_df = pd.DataFrame({'ds': ds, 'y': values})
            
            m = Prophet(
                seasonality_mode='multiplicative',
                changepoint_prior_scale=0.3,
                yearly_seasonality=False,
                weekly_seasonality=False,
                daily_seasonality=False
            )
            m.fit(prophet_df)
            
            # Forecast
            future = m.make_future_dataframe(periods=forecast_periods)
            forecast = m.predict(future)
            
            # Extract trend slope (linear approx over last + forecast)
            trend_vals = forecast['trend'].values
            n = len(trend_vals)
            slope = (trend_vals[-1] - trend_vals[n // 2]) / max(n // 2, 1)
            
            forecast_result = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'trend']].copy()
            forecast_result['cycle'] = range(1, len(forecast_result) + 1)
            
            return {
                'forecast_df': forecast_result,
                'trend_slope': float(slope),
                'accelerating_degradation': abs(slope) > 0.5
            }
        except Exception as e:
            print(f"      [Prophet Error] {e}. Using fallback.")
    
    # ---- NUMPY FALLBACK: Linear trend + residual analysis ----
    x = np.arange(len(values))
    coeffs = np.polyfit(x, values, deg=2)  # Quadratic fit for acceleration detection
    
    # Linear slope
    linear_coeffs = np.polyfit(x, values, deg=1)
    trend_slope = linear_coeffs[0]
    
    # Acceleration = 2nd derivative (coefficient of x^2)
    acceleration = 2 * coeffs[0]
    
    # Generate forecast
    future_x = np.arange(len(values) + forecast_periods)
    forecast_vals = np.polyval(coeffs, future_x)
    
    # Simple confidence bands (±2 std of residuals)
    residuals = values - np.polyval(coeffs, x)
    std_resid = np.std(residuals)
    
    forecast_df = pd.DataFrame({
        'cycle': future_x + 1,
        'yhat': forecast_vals,
        'yhat_lower': forecast_vals - 2 * std_resid,
        'yhat_upper': forecast_vals + 2 * std_resid,
        'trend': np.polyval(linear_coeffs, future_x)
    })
    
    return {
        'forecast_df': forecast_df,
        'trend_slope': float(trend_slope),
        'accelerating_degradation': abs(acceleration) > 0.01
    }
