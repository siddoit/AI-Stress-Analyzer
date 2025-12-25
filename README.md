# 🧠 NeuralHealth: AI Stress & Ergonomic Analyzer

![Python](https://img.shields.io/badge/Python-3.10%2B-blue) ![TensorFlow](https://img.shields.io/badge/TensorFlow-GPU-orange) ![Status](https://img.shields.io/badge/Status-Active-green)

**NeuralHealth** is a real-time desktop application designed to monitor ergonomic health and stress levels during computer usage. By combining computer vision for posture and emotion detection with external environmental sensors (ESP32), the system provides a comprehensive "Stress Score" to help users maintain a healthy work environment. 

The application utilizes multi-threading to ensure high performance, running complex AI inference tasks on a background thread so the graphical interface remains smooth and responsive.

## ✨ Features

* **Real-Time Posture Tracking:** Instantly detects "Slouching" or "Leaning" and provides immediate visual feedback.
* **Emotion Analysis:** Analyzes facial micro-expressions to detect stress, anger, or fatigue using the `enet_b0` model.
* **Environmental Monitoring:** Connects to an ESP32 to track **Room Temp, Humidity, and Noise levels**.
* **Smart Scoring:** Fuses posture, emotion, and environment data into a single 0-100% Stress Score.
* **Turbo-Threaded Backend:** Features a dedicated background thread for heavy AI math, ensuring the UI never lags or freezes.
* **Simulation Mode:** Automatically switches to simulated data if no hardware is detected.

---

## 📂 Project Structure

```text
NeuralHealth/
│
├── README.md               
├── launch.py               <-- Main Application Launcher
├── requirements.txt        <-- Dependencies list
│
├── models/                 
│   ├── enet_b0_8_best_vgaf.pt
│   └── my_posture_model.h5
│
└── src/                    
    ├── db.py               <-- Database Handler
    ├── emo.py              <-- Emotion Logic
    ├── posture_check.py    <-- Posture Logic
    └── trainer.py          <-- Training Script
```

## 🛠️ Prerequisites

Before running the system, ensure you have:

* **Python 3.9+**: Installed and added to your system PATH.
* **Webcam**: Required for posture and emotion detection.
* **ESP32 (Optional)**: For environmental sensing (Temp/Hum/Noise).

---

## 🚀 Installation & Setup

### 1. Initial Setup
* Navigate to the `NeuralHealth` folder.
* Install the required libraries:
    ```bash
    pip install -r requirements.txt
    ```

### 2. Device Setup (Hardware)
* Connect your ESP32 to the computer via USB.
* Open `launch.py` in a text editor.
* Find the line `ARDUINO_PORT = 'COM10'` and change it to your device's port (e.g., `COM3`).
    > **Note:** If you do not have an ESP32, you can skip this. The system will auto-detect the missing device and enter "Simulation Mode."

---

## 🖥️ How to Run

* Open your terminal in the project folder.
* Run the launcher:
    ```bash
    python launch.py
    ```
* Wait for the **NeuralHealth Dashboard** to open.
* **Sit back**: The system will calibrate and start tracking your posture and stress levels immediately.

---

## ⚙️ Usage Guide

* **Posture Correction:** If the text turns **RED** and says "Slouching," sit up straight. The text will turn **GREEN** ("Good Posture").
* **Stress Score:**
    * **0-40% (Green):** Optimal State.
    * **40-75% (Yellow):** Moderate Stress.
    * **75-100% (Red):** Critical Stress. Take a break.
* **Environment:** If the noise level exceeds 65dB, the system will flag it as a stress factor.

---

## ⚠️ Troubleshooting

**The app is laggy or slow.**
* Ensure you are using the "Turbo" version (Multi-threaded).
* Check if another app is using your webcam.

**"Connection Failed" for ESP32.**
* Check if the `ARDUINO_PORT` in `launch.py` matches your Device Manager.
* Ensure the Baud Rate is set to `115200`.

**Camera opens but closes immediately.**
* Verify that `models/enet_b0_8_best_vgaf.pt` exists. If not, the system cannot load the model.

---

**Built with:** Python, CustomTkinter, MediaPipe, TensorFlow, & ESP32.