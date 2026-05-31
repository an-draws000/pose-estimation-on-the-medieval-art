import numpy as np
import pickle
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor

# Load pairs saved from find_pairs.py
X_yolo = np.load("X_yolo.npy")
X_manual = np.load("X_manual.npy")
y = np.load("y.npy", allow_pickle=True)  # allow_pickle=True because y contains strings

def train_joint_regressors(X_yolo, X_manual, alpha=10.0):
    """Train one regressor per joint."""
    X_yolo_3d = X_yolo.reshape(-1, 17, 2)
    X_manual_3d = X_manual.reshape(-1, 17, 2)
    
    regressors = {}
    for j in range(17):  # all joints
        x_in = X_yolo_3d[:, j, :]    # (100, 2)
        x_out = X_manual_3d[:, j, :] # (100, 2)
        
        # Use all other joints as context
        context = np.delete(X_yolo_3d, j, axis=1).reshape(len(X_yolo), -1)  # (100, 32)
        x_in_full = np.hstack([x_in, context])  # (100, 34)
        
        reg = Ridge(alpha=alpha)
        reg.fit(x_in_full, x_out)
        regressors[j] = reg
    
    return regressors


regressors = train_joint_regressors(X_yolo, X_manual, alpha=10.0)

from sklearn.model_selection import KFold

kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Baseline: how far is raw YOLO from manual
baseline = np.mean(np.linalg.norm(
    X_yolo.reshape(-1, 17, 2) - X_manual.reshape(-1, 17, 2), axis=2))
print(f"Baseline YOLO error: {baseline:.4f}")

fold_errors = []
for train_idx, test_idx in kf.split(X_yolo):
    regressors = train_joint_regressors(X_yolo[train_idx], X_manual[train_idx])
    
    # Correct test set
    X_yolo_3d = X_yolo[test_idx].reshape(-1, 17, 2)
    X_corrected = np.zeros_like(X_yolo_3d)
    
    for j in range(17):
        context = np.delete(X_yolo_3d, j, axis=1).reshape(len(test_idx), -1)
        x_in = np.hstack([X_yolo_3d[:, j, :], context])
        X_corrected[:, j, :] = regressors[j].predict(x_in)
    
    error = np.mean(np.linalg.norm(
        X_corrected - X_manual[test_idx].reshape(-1, 17, 2), axis=2))
    fold_errors.append(error)

print(f"Corrected error:     {np.mean(fold_errors):.4f} ± {np.std(fold_errors):.4f}")
print(f"Improvement:         {baseline - np.mean(fold_errors):.4f}")


with open("joint_regressors.pkl", "wb") as f:
    pickle.dump(regressors, f)

print("Regressor saved.")

