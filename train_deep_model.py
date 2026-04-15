import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping
import joblib

# -----------------------------
# LOAD DATA
# -----------------------------
X = np.load("X.npy")
y = np.load("y.npy")

print("Data shape:", X.shape)
print("Class distribution:", np.bincount(y))

# -----------------------------
# 🔥 NORMALIZATION (VERY IMPORTANT)
# -----------------------------
scaler = StandardScaler()
X = scaler.fit_transform(X)

# save scaler for prediction
joblib.dump(scaler, "scaler.pkl")

# -----------------------------
# TRAIN-TEST SPLIT
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# MODEL
# -----------------------------
model = Sequential()

model.add(Dense(256, activation="relu", input_shape=(10,)))
model.add(Dense(128, activation="relu"))
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(32, activation="relu"))
model.add(Dense(2, activation="softmax"))  # ✅ only ONE output layer

# -----------------------------
# COMPILE
# -----------------------------
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# -----------------------------
# 🔥 EARLY STOPPING
# -----------------------------
early = EarlyStopping(
    patience=10,
    restore_best_weights=True
)

# -----------------------------
# TRAIN
# -----------------------------
model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=16,
    validation_data=(X_test, y_test),
    callbacks=[early]
)
# -----------------------------
# 🔥 EVALUATE MODEL
# -----------------------------
loss, acc = model.evaluate(X_test, y_test)
print("Final Test Accuracy:", acc)
# -----------------------------
# SAVE MODEL
# -----------------------------
model.save("brain_model.h5")

print("✅ Model trained successfully")