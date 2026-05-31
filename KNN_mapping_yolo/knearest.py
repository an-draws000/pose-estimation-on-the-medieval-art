from sklearn.model_selection import KFold
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
import pickle

X_yolo = np.load("X_yolo.npy")
X_manual = np.load("X_manual.npy")
kf = KFold(n_splits=5, shuffle=True, random_state=42)

error_yolo = np.mean(np.linalg.norm(
    X_yolo.reshape(-1, 17, 2) - X_manual.reshape(-1, 17, 2), axis=2))
print(f"YOLOv8 baseline error: {error_yolo:.4f}")

# for k in [3, 5, 7, 10]:
# for k in [10, 15, 20, 25, 30]:
# for k in [20, 35, 40, 50]:
#     fold_errors = []
#     for train_idx, test_idx in kf.split(X_yolo):
#         knn = KNeighborsRegressor(n_neighbors=k, weights='distance')
#         knn.fit(X_yolo[train_idx], X_manual[train_idx])
        
#         X_corrected = knn.predict(X_yolo[test_idx])
#         error = np.mean(np.linalg.norm(
#             X_corrected.reshape(-1, 17, 2) - X_manual[test_idx].reshape(-1, 17, 2), axis=2))
#         fold_errors.append(error)
    
#     print(f"KNN k={k} CV error: {np.mean(fold_errors):.4f} ± {np.std(fold_errors):.4f}")

fold_errors = []
for train_idx, test_idx in kf.split(X_yolo):
    knn = KNeighborsRegressor(n_neighbors=20, weights='distance')
    knn.fit(X_yolo[train_idx], X_manual[train_idx])
    
    X_corrected = knn.predict(X_yolo[test_idx])
    error = np.mean(np.linalg.norm(
        X_corrected.reshape(-1, 17, 2) - X_manual[test_idx].reshape(-1, 17, 2), axis=2))
    fold_errors.append(error)

print(f"Corrected error:     {np.mean(fold_errors):.4f} ± {np.std(fold_errors):.4f}")
print(f"Improvement:         {error_yolo - np.mean(fold_errors):.4f}")

knn = KNeighborsRegressor(n_neighbors=20, weights='distance')
knn.fit(X_yolo, X_manual)

with open("joint_knn.pkl", "wb") as f:
    pickle.dump(knn, f)

print("KNN saved with k=20")