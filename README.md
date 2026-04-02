# NeuralHealth: AI Edge-Compute Stress & Ergonomic Analyzer

![Python](https://img.shields.io/badge/Python-3.12-blue) ![Flask](https://img.shields.io/badge/Flask-Web_App-lightgrey) ![TensorFlow](https://img.shields.io/badge/TensorFlow-GPU-orange) ![Status](https://img.shields.io/badge/Status-Active-green)

**NeuralHealth** is a real-time, edge-AI web dashboard designed to monitor ergonomic health, environmental conditions, and cognitive stress during intense workflow sessions. 

Transitioning from a heavy desktop GUI to a lightweight Flask-based web architecture, this system fuses computer vision (posture and facial micro-expressions) with external hardware telemetry (ESP32) to generate a live "Net Stress Load." If you hit a critical burnout threshold, the system triggers local Windows interrupts and fires off Telegram alerts to step you away from the desk.

## 🚀 Core Features

* **Zero-Latency Web UI:** A premium, dark-mode CSS Grid dashboard that renders your camera feed and telemetry in real-time.
* **DirectShow Camera Pipeline:** Bypasses Windows buffering for absolute bleeding-edge frame acquisition without the classic OpenCV lag.
* **Asynchronous AI Engine:** Heavy ML math (MediaPipe for posture, HSEmotion for cognitive state) runs on an isolated background thread to guarantee the UI never freezes.
* **Hardware Telemetry Fusion:** Connects seamlessly to an ESP32 micro-controller to track **Room Temp, Humidity, Noise, and Light (lx)**.
* **Burnout Protocols:** Automatically triggers system buzzers, Windows UI interrupts, and Telegram notifications when critical stress is sustained.
* **Single-Click Boot:** An automated batch script handles virtual environments, massive AI dependencies, and browser launching without manual terminal commands.

---

## 📂 Architecture & Structure

```text
NeuralHealth/
│
├── installation.bat       <-- The single-click setup and boot script
├── run.py                 <-- Threaded launcher (Spins up Flask & opens browser)
├── app.py                 <-- Flask routing and web server
├── logic.py               <-- The Brains: AI inference, OpenCV, Serial comms
├── requirements.txt       <-- Heavily optimized dependency list
├── .env                   <-- API Keys and Secrets (Local only)
│
├── templates/             
│   └── index.html         <-- The Frontend Dashboard
│
├── Models/                <-- Local ML Weights
│   ├── enet_b0_8_best_vgaf.pt
│   └── my_posture_model.h5
│
└── src/                   <-- Legacy / Database utilities

🛠️ Prerequisites
-----------------

*   **Python 3.12+**: The installation.bat will attempt to install this automatically if you don't have it.
    
*   **Webcam**: Required for real-time spatial and emotional mapping.
    
*   **ESP32 (Optional)**: For environmental hardware sensing. The system gracefully degrades to "Simulation Mode" if unplugged.
    
*   **Telegram Account**: To receive critical load alerts.
    

🔐 Setup: Telegram Bot Integration
----------------------------------

To get the automated mobile alerts working, you need to set up a Telegram Bot and grab your API credentials. Do not skip this if you want the full experience.

1.  **Create the Bot:**
    
    *   Open Telegram and search for **@BotFather**.
        
    *   Send the command /newbot and follow the prompts to name it.
        
    *   BotFather will give you an **HTTP API Token** (e.g., 123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11). Copy this.
        
2.  **Get Your Chat ID:**
    
    *   Search for **@RawDataBot** or **@userinfobot** in Telegram and press Start.
        
    *   It will output a JSON block. Look for the "chat": {"id": 123456789} line. Copy that number.
        
3.  **Configure the Environment:**
    
    *   Create a file named exactly .env in the root folder of this project.
        
    *   Code snippetTELEGRAM\_BOT\_TOKEN=paste\_your\_token\_hereTELEGRAM\_CHAT\_ID=paste\_your\_chat\_id\_here
        

⚡ Installation & Execution
--------------------------

We have completely eliminated the need for manual setup.

1.  **Clone/Download** this repository.
    
2.  **Double-click installation.bat**.
    

**What the script does automatically:**

*   Checks if Python is installed (and downloads it silently if missing).
    
*   Builds a highly isolated Virtual Environment (venv).
    
*   Installs gigabytes of AI models and dependencies (grab a coffee on the first run).
    
*   Drops a .installed marker so it skips the download phase on future boots.
    
*   Spins up the Flask server and instantly opens your default browser to http://127.0.0.1:5000.
    

📊 Dashboard Guide
------------------

*   **Atmospherics Panel:** Tracks your room's physical condition. You can toggle "Laptop Mic" if you want to bypass the ESP32 for decibel readings.
    
*   **Live Stress Trend:** A rolling 30-tick graph showing your real-time cognitive load.
    
*   **Care Diagnostics:**
    
    *   **Optimal State (Green):** You are locked in. Posture is solid, emotion is neutral/happy.
        
    *   **Burnout Risk (Orange):** You are slouching, leaning, or the environment is hostile (too loud, too dark). System issues ergonomic corrections.
        
    *   **Critical Load (Red):** Sustained anger/stress combined with physical degradation. Triggers the Telegram alert pipeline and hardware buzzers to force a break.
        

🐛 Troubleshooting
------------------

*   **"AttributeError: 'StressEngine' object has no attribute 'running'"**Ensure you are using the latest logic.py where threads are started inside the start() method, not \_\_init\_\_().
    
*   **Camera feed keeps expanding and breaking the UI.**Ensure your index.html has minmax(0, 1fr) set in the CSS Grid template columns.
    
*   **Camera opens but the app crashes instantly.**Ensure your weights inside the Models/ folder actually exist and downloaded properly. The engine cannot boot without them.
    
*   **No Telegram alerts are firing.**Check your .env file. Ensure you actually messaged your newly created bot at least once so it has permission to send you messages.