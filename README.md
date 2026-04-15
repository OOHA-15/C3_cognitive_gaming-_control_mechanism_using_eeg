# EEG-Based Cognitive Gaming Control System

## 📖 Description
This project presents a Brain-Computer Interface (BCI) based system that allows users to control a game using their brain signals. The system captures EEG signals from the user through electrodes and processes them to identify mental states such as attentive and relaxed. Based on the detected state, the system controls game actions in real-time without the need for traditional input devices like a keyboard or mouse.

---

## 🎯 Objective
The main objective of this project is to develop a system that enables hands-free interaction by using brain signals. It aims to create a simple and effective method for controlling a game based on the user’s level of focus.

---

## ⚙️ Implementation Details

### 1. Data Collection
EEG signals are collected using electrodes connected to a BIO AMP (EXG Pill) and Arduino. The signals are transmitted to the system through serial communication and stored for further processing.

### 2. Signal Preprocessing
The raw EEG signals are filtered using techniques such as:
- Moving Average (for smoothing)
- Low-pass filtering (to remove noise)

This ensures that the signal is clean and suitable for analysis.

### 3. Feature Extraction
From the processed signal, important features are extracted such as:
- Mean
- Standard Deviation
- Variance
- Energy
- Peak values

These features help in understanding the behavior of the signal.

### 4. Model Training
The extracted features are used to train a classification model. The model learns to distinguish between attentive and relaxed states based on the input data.

### 5. Real-Time Prediction
The system continuously reads incoming EEG signals, processes them in small segments, extracts features, and predicts the mental state in real time.

### 6. Game Control
Based on the prediction:
- Attentive → Car moves forward
- Relaxed → Car stops

This enables smooth and interactive gameplay using brain signals.

---

## 🛠️ Technologies Used
- Python
- NumPy, Pandas, SciPy
- Scikit-learn
- TensorFlow / Keras
- PySerial
- PyAutoGUI

---

## 💻 Hardware Used
- EEG Electrodes
- BIO AMP / EXG Pill
- Arduino
- Computer / Laptop

---

## 🚀 Output
The system provides real-time control of a game based on the user’s mental state, creating an interactive and hands-free gaming experience.

---

## 📌 Future Scope
The system can be enhanced to support more control actions, improve accuracy, and be extended to real-world applications such as assistive technologies.
