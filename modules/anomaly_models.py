"""
ERR-CC v4.0 — Anomaly Detection Models
=======================================
Contains:
  - FraudAutoencoder (PyTorch): Fully-connected AE for reconstruction-error anomaly detection
  - GNNLiteFraudScorer: Manual 1-hop neighborhood aggregation + XGBoost classifier
  
All PyTorch code is wrapped in try/except for graceful fallback.
"""

import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings('ignore')

# --- PyTorch availability check ---
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("  [WARNING] PyTorch not available. Autoencoder will use sklearn fallback.")

from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb


# ============================================================
# 1. FRAUD AUTOENCODER (PyTorch)
# ============================================================

if TORCH_AVAILABLE:
    class FraudAutoencoder(nn.Module):
        """
        Fully-connected autoencoder for unsupervised anomaly detection.
        Architecture: Input → 64 → 32 → 16 (bottleneck) → 32 → 64 → Input
        Anomaly score = per-sample reconstruction MSE.
        """
        def __init__(self, input_dim):
            super(FraudAutoencoder, self).__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 16)  # bottleneck
            )
            self.decoder = nn.Sequential(
                nn.Linear(16, 32),
                nn.ReLU(),
                nn.Linear(32, 64),
                nn.ReLU(),
                nn.Linear(64, input_dim)
            )

        def forward(self, x):
            encoded = self.encoder(x)
            decoded = self.decoder(encoded)
            return decoded


def train_autoencoder(X_normal, epochs=30, lr=1e-3, batch_size=256):
    """
    Train the autoencoder on NORMAL transactions only.
    
    Args:
        X_normal: numpy array of normal transaction features (isFraud == 0)
        epochs: number of training epochs
        lr: learning rate for Adam optimizer
        batch_size: mini-batch size
    
    Returns:
        model: trained FraudAutoencoder
        scaler: fitted MinMaxScaler for normalizing reconstruction errors
        final_loss: final training loss value
    """
    if not TORCH_AVAILABLE:
        print("    [FALLBACK] PyTorch unavailable. Returning None.")
        return None, None, 0.0

    torch.manual_seed(42)
    np.random.seed(42)
    
    input_dim = X_normal.shape[1]
    model = FraudAutoencoder(input_dim)
    
    # Convert to tensors
    X_tensor = torch.FloatTensor(X_normal)
    dataset = TensorDataset(X_tensor, X_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    model.train()
    final_loss = 0.0
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
        final_loss = epoch_loss / max(n_batches, 1)
        if (epoch + 1) % 10 == 0:
            print(f"      Epoch {epoch+1}/{epochs} — Loss: {final_loss:.6f}")
    
    # Compute reconstruction errors on training data for scaler fitting
    model.eval()
    with torch.no_grad():
        recon = model(X_tensor)
        errors = torch.mean((X_tensor - recon) ** 2, dim=1).numpy()
    
    scaler = MinMaxScaler()
    scaler.fit(errors.reshape(-1, 1))
    
    return model, scaler, final_loss


def ae_anomaly_scores(model, X, scaler):
    """
    Compute normalized anomaly scores using the trained autoencoder.
    
    Args:
        model: trained FraudAutoencoder
        X: numpy array of all transaction features
        scaler: fitted MinMaxScaler from training
    
    Returns:
        scores: numpy array of normalized anomaly scores in [0, 1]
    """
    if not TORCH_AVAILABLE or model is None:
        return np.zeros(len(X))
    
    model.eval()
    X_tensor = torch.FloatTensor(X)
    
    with torch.no_grad():
        recon = model(X_tensor)
        errors = torch.mean((X_tensor - recon) ** 2, dim=1).numpy()
    
    # Normalize to [0, 1]
    scores = scaler.transform(errors.reshape(-1, 1)).flatten()
    scores = np.clip(scores, 0, 1)
    return scores


# ============================================================
# 2. GNN-LITE FRAUD SCORER 
#    (Manual 1-hop neighborhood aggregation + XGBoost)
# ============================================================

class GNNLiteFraudScorer:
    """
    GNN-inspired fraud scorer without requiring torch_geometric.
    
    Strategy:
    1. For each node, aggregate MEAN of neighbor features from adjacency list
    2. Concatenate [node_features, aggregated_neighbor_features] → 2x feature dim
    3. Train XGBoost classifier on concatenated features with isFraud labels
    4. Output risk scores per node
    """
    
    def __init__(self):
        self.model = None
        self.feature_cols = None
    
    def _aggregate_neighbors(self, G, node_features_df, feature_cols):
        """
        For each node, compute mean of neighbor features (1-hop aggregation).
        This is the core GNN operation done manually.
        """
        agg_data = []
        nodes = list(node_features_df.index)
        
        for node in nodes:
            if G.has_node(node):
                neighbors = list(G.predecessors(node)) + list(G.successors(node))
                neighbors = [n for n in neighbors if n in node_features_df.index]
                
                if len(neighbors) > 0:
                    # Mean aggregation of neighbor features
                    neighbor_feats = node_features_df.loc[neighbors, feature_cols].mean().values
                else:
                    neighbor_feats = np.zeros(len(feature_cols))
            else:
                neighbor_feats = np.zeros(len(feature_cols))
            
            # Concatenate: [own_features, neighbor_mean_features]
            own_feats = node_features_df.loc[node, feature_cols].values
            combined = np.concatenate([own_feats, neighbor_feats])
            agg_data.append(combined)
        
        col_names = (
            [f"own_{c}" for c in feature_cols] + 
            [f"neigh_{c}" for c in feature_cols]
        )
        return pd.DataFrame(agg_data, index=nodes, columns=col_names)
    
    def train(self, G, node_features_df, labels_series, feature_cols):
        """
        Train the GNN-lite scorer.
        
        Args:
            G: NetworkX DiGraph
            node_features_df: DataFrame indexed by node ID with graph features
            labels_series: Series indexed by node ID with binary fraud labels
            feature_cols: list of feature column names to use
        """
        self.feature_cols = feature_cols
        
        # Aggregate neighbor features
        X_agg = self._aggregate_neighbors(G, node_features_df, feature_cols)
        
        # Align labels with available nodes
        common_nodes = X_agg.index.intersection(labels_series.index)
        X_train = X_agg.loc[common_nodes]
        y_train = labels_series.loc[common_nodes]
        
        X_train = X_train.replace([np.inf, -np.inf], 0).fillna(0)
        
        if len(y_train) == 0 or y_train.sum() == 0:
            print("    [WARNING] No positive labels for GNN-lite training. Using dummy model.")
            self.model = None
            return
        
        # Train XGBoost on concatenated features
        fraud_ratio = max((y_train == 0).sum() / max((y_train == 1).sum(), 1), 1)
        self.model = xgb.XGBClassifier(
            n_estimators=100, max_depth=5, learning_rate=0.1,
            scale_pos_weight=fraud_ratio, eval_metric='aucpr',
            random_state=42, use_label_encoder=False, verbosity=0
        )
        self.model.fit(X_train, y_train)
        print(f"      GNN-lite trained on {len(X_train)} nodes ({int(y_train.sum())} positive)")
    
    def predict(self, G, node_features_df):
        """
        Predict risk scores for all nodes.
        
        Returns:
            Series indexed by node ID with risk scores in [0, 1]
        """
        if self.model is None or self.feature_cols is None:
            return pd.Series(0.0, index=node_features_df.index)
        
        X_agg = self._aggregate_neighbors(G, node_features_df, self.feature_cols)
        X_agg = X_agg.replace([np.inf, -np.inf], 0).fillna(0)
        
        scores = self.model.predict_proba(X_agg)[:, 1]
        return pd.Series(scores, index=X_agg.index)
