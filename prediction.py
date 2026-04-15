import numpy as np
import serial
import pyautogui
import joblib
from tensorflow.keras.models import load_model
from scipy.signal import butter, filtfilt

WINDOW = 256
THRESHOLD = 0.45

# -----------------------------
# 🔥 LOAD MODEL + SCALER
# -----------------------------
model = load_model("brain_model.h5")
scaler = joblib.load("scaler.pkl")

ser = serial.Serial("COM9", 115200, timeout=1)

buffer = []
pred_history = []

print("✅ Prediction started (Final Version)...")

# -----------------------------
# PREPROCESSING
# -----------------------------
def moving_average(signal, window=5):
    return np.convolve(signal, np.ones(window)/window, mode='same')

def low_pass_filter(signal, cutoff=3, fs=50, order=3):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low')
    return filtfilt(b, a, signal)

# -----------------------------
# MAIN LOOP
# -----------------------------
while True:
    try:
        line = ser.readline().decode().strip()

        if not line:
            continue

        value = float(line)
        buffer.append(value)

        print("Raw:", value)

        if len(buffer) >= WINDOW:

            segment = np.array(buffer[-WINDOW:])

            # -----------------------------
            # 🔥 PREPROCESSING
            # -----------------------------
            segment = moving_average(segment)
            segment = low_pass_filter(segment)
            segment = (segment - np.mean(segment)) / (np.std(segment) + 1e-6)

            # -----------------------------
            # 🔥 FEATURE EXTRACTION
            # -----------------------------
            mean = np.mean(segment)
            std = np.std(segment)
            var = np.var(segment)
            mx = np.max(segment)
            mn = np.min(segment)
            rng = mx - mn

            diff = np.diff(segment)
            diff_mean = np.mean(diff)
            diff_std = np.std(diff)

            energy = np.sum(segment**2)
            peaks = np.sum(segment > (mean + std))

            features = np.array([
                mean, std, var, mx, mn, rng,
                diff_mean, diff_std,
                energy, peaks
            ]).reshape(1, 10)

            # -----------------------------
            # 🔥 APPLY SCALER (VERY IMPORTANT)
            # -----------------------------
            features = scaler.transform(features)

            # -----------------------------
            # 🔥 PREDICTION
            # -----------------------------
            pred = model.predict(features, verbose=0)
            print("Prediction:", pred)
            focus_prob = pred[0][1]

            # smoothing
            pred_history.append(focus_prob)
            if len(pred_history) > 10:
                pred_history.pop(0)

            avg_prob = sum(pred_history) / len(pred_history)

            print(f"Focus Prob: {focus_prob:.3f} | Avg: {avg_prob:.3f}")

            # -----------------------------
            # 🚗 CONTROL
            # -----------------------------
            if avg_prob > THRESHOLD:
                pyautogui.keyDown("w")
                print("🚗 Focus → Car Moving")
            else:
                pyautogui.keyUp("w")
                print("🛑 Relax → Car Stopped")

    except Exception as e:
        print("Error:", e)