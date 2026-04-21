import numpy as np
from sklearn.neighbors import KNeighborsClassifier

# Dataset (Salary, Debt)
X = np.array([
    [3,8],
    [4,7],
    [5,6],
    [6,5],
    [7,4],
    [8,3],
    [9,2],
    [10,2],
    [2,9],
    [11,1]
])

# Label (0=Eligible, 1=Not Eligible)
y = np.array([
    1,  # [3,8]
    1,  # [4,7]
    1,  # [5,6]
    0,  # [6,5]
    0,  # [7,4]
    0,  # [8,3]
    0,  # [9,2]
    0,  # [10,2]
    1,  # [2,9]
    0   # [11,1]
])

# Model KNN
knn = KNeighborsClassifier(n_neighbors=3, metric='euclidean')

# Training
knn.fit(X, y)

# Data baru
new_data = np.array([[10,3]])

# Prediksi
prediction = knn.predict(new_data)

# Convert ke label
label_map = {0: "Eligible", 1: "Not Eligible"}

print("Prediction (numeric):", prediction[0])
print("Prediction (label):", label_map[prediction[0]])
