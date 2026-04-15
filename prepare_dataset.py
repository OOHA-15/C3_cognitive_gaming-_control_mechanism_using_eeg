import pandas as pd
import numpy as np
from sklearn.utils import shuffle, resample

# -----------------------------
WINDOW = 256
STEP = 50

# -----------------------------
# LOAD DATA
# -----------------------------
data = pd.read_csv("final_dataset.csv")

# 🔥 speed up (important)
data = data.sample(n=50000, random_state=42)

values = data["value"].values
labels = data["label"].values

# -----------------------------
# FEATURE EXTRACTION
# -----------------------------
X = []
y = []

for i in range(0, len(values) - WINDOW, STEP):

    # progress print (so it doesn't feel stuck)
    if i % 10000 == 0:
        print("Processing:", i)

    seg = values[i:i+WINDOW]

    # -----------------------------
    # 🔥 STRONG FEATURES
    # -----------------------------
    mean = np.mean(seg)
    std = np.std(seg)
    var = np.var(seg)
    mx = np.max(seg)
    mn = np.min(seg)
    rng = mx - mn

    diff = np.diff(seg)
    diff_mean = np.mean(diff)
    diff_std = np.std(diff)

    energy = np.sum(seg**2)
    peaks = np.sum(seg > (mean + std))

    X.append([
        mean, std, var, mx, mn, rng,
        diff_mean, diff_std,
        energy, peaks
    ])

    # -----------------------------
    # ✅ MAJORITY LABEL
    # -----------------------------
    window_labels = labels[i:i+WINDOW]
    label = round(np.mean(window_labels))
    y.append(label)

# -----------------------------
# CONVERT
# -----------------------------
X = np.array(X)
y = np.array(y)

print("Before balancing:", np.bincount(y))

# -----------------------------
# 🔥 BALANCE DATASET (UPSAMPLING)
# -----------------------------
X_relaxed = X[y == 0]
X_attentive = X[y == 1]

y_relaxed = y[y == 0]
y_attentive = y[y == 1]

# upsample attentive
X_attentive_up, y_attentive_up = resample(
    X_attentive, y_attentive,
    replace=True,
    n_samples=len(X_relaxed),
    random_state=42
)

# combine
X = np.vstack((X_relaxed, X_attentive_up))
y = np.hstack((y_relaxed, y_attentive_up))

# shuffle
X, y = shuffle(X, y, random_state=42)

print("✅ Balanced distribution:", np.bincount(y))

# -----------------------------
# SAVE
# -----------------------------
np.save("X.npy", X)
np.save("y.npy", y)

print("✅ Dataset ready")
print("Shape:", X.shape)