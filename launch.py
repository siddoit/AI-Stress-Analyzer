import customtkinter as ctk
import cv2
import torch
import numpy as np
import mediapipe as mp
import tensorflow as tf
import os
import shutil
import statistics
import serial          # Reads your ESP32
from pathlib import Path
from PIL import Image, ImageTk
from collections import deque
import threading
import time

# --- CONFIGURATION ---
# IMPORTANT: Check Device Manager for your ESP32 Port!
ARDUINO_PORT = 'COM10'  
# Your ESP32 code uses 115200, so Python MUST match it
BAUD_RATE = 115200     

HISTORY_LEN = 5
SKIP_FRAMES = 3 

# System Fixes for Tensorflow/PyTorch
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
_original_load = torch.load
def fixed_load(*args, **kwargs):
    if 'weights_only' not in kwargs: kwargs['weights_only'] = False
    return _original_load(*args, **kwargs)
torch.load = fixed_load
from hsemotion.facial_emotions import HSEmotionRecognizer

# ==========================================
# 🧠 BACKEND CLASS (THREADED)
# ==========================================
class StressBrain:
    def __init__(self):
        # --- SHARED VARIABLES (The GUI reads these) ---
        self.latest_pos = "Good Posture"
        self.latest_emo = "Neutral"
        self.latest_noise = 0
        self.latest_temp = 25.0
        self.latest_hum = 50.0
        self.frame_to_process = None # The GUI puts frames here
        self.running = True
        
        # --- HARDWARE SETUP ---
        self.ser = None
        try:
            self.ser = serial.Serial(ARDUINO_PORT, BAUD_RATE, timeout=0.1)
            print(f"✅ Hardware: ESP32 Connected on {ARDUINO_PORT}")
        except:
            print(f"⚠️ Hardware: Connection failed (Simulating Data)")

        # --- AI SETUP ---
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Load Emotion (Your original code)
        local_pt = 'enet_b0_8_best_vgaf.pt'
        cache_path = os.path.join(str(Path.home()), '.hsemotion', local_pt)
        if not os.path.exists(cache_path) and os.path.exists(local_pt):
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            shutil.copy2(local_pt, cache_path)
            
        try:
            self.emo_model = HSEmotionRecognizer(model_name='enet_b0_8_best_vgaf', device=self.device)
        except: self.emo_model = None

        # Load Posture
        try:
            self.posture_model = tf.keras.models.load_model('my_posture_model.h5')
            self.pos_classes = ['Good', 'Slouching', 'Leaning']
        except: self.posture_model = None

        self.mp_face = mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)
        self.mp_pose = mp.solutions.pose.Pose(model_complexity=0, min_detection_confidence=0.5)
        self.pos_hist = deque(maxlen=HISTORY_LEN)

    def start(self):
        """Starts the background worker thread"""
        t = threading.Thread(target=self._worker_loop)
        t.daemon = True # Kills thread when app closes
        t.start()

    def _worker_loop(self):
        """This loop runs parallel to the GUI. No lag allowed here."""
        while self.running:
            # 1. READ HARDWARE (This used to block your GUI!)
            if self.ser and self.ser.in_waiting > 0:
                try:
                    line = self.ser.readline().decode('utf-8').strip()
                    if "Sound:" in line and "Temp:" in line:
                        parts = line.split('|')
                        self.latest_noise = float(parts[0].split(':')[1].replace('dB', '').strip())
                        self.latest_temp = float(parts[1].split(':')[1].replace('C', '').strip())
                        if len(parts) > 2:
                            self.latest_hum = float(parts[2].split(':')[1].strip())
                except: pass

            # 2. PROCESS AI (Only if there is a new frame waiting)
            if self.frame_to_process is not None:
                # Grab the frame and clear the slot immediately
                curr_frame = self.frame_to_process.copy() 
                self.frame_to_process = None 
                
                self._run_inference(curr_frame)
            
            # Tiny sleep to prevent CPU roasting
            time.sleep(0.01)

    def _run_inference(self, frame):
        # --- POSTURE ---
        results = self.mp_pose.process(frame)
        if results.pose_landmarks and self.posture_model:
            lm = []
            for i in range(17):
                p = results.pose_landmarks.landmark[i]
                lm.extend([p.x, p.y, p.z, p.visibility])
            pred = self.posture_model.predict(np.array(lm).reshape(1,-1), verbose=0)
            pred[0][0] += 0.15 # Bias fix
            
            self.pos_hist.append(np.argmax(pred))
            try: idx = statistics.mode(self.pos_hist)
            except: idx = np.argmax(pred)
            self.latest_pos = self.pos_classes[idx]

        # --- EMOTION ---
        f_res = self.mp_face.process(frame)
        if f_res.detections and self.emo_model:
            d = f_res.detections[0]
            bb = d.location_data.relative_bounding_box
            H, W, _ = frame.shape
            x, y, w, h = int(bb.xmin*W), int(bb.ymin*H), int(bb.width*W), int(bb.height*H)
            if w > 20:
                try:
                    # Fix: Ensure coordinates are within bounds
                    f_img = frame[max(0,y):min(H,y+h), max(0,x):min(W,x+w)]
                    self.latest_emo, _ = self.emo_model.predict_emotions(f_img, logits=False)
                except: pass


# ==========================================
# 🖥️ FRONTEND (THE FIX)
# ==========================================
class StressApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        # 1. Start the Brain (Background Thread)
        self.brain = StressBrain()
        self.brain.start() 

        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, 640)
        self.cap.set(4, 480)
        
        # --- UI LAYOUT ---
        self.title("AI Ergonomic Stress Analyzer (Turbo)")
        self.geometry("1100x700")
        ctk.set_appearance_mode("Dark")
        ctk.set_default_color_theme("blue")

        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Sidebar
        self.sidebar = ctk.CTkFrame(self, width=220, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        
        ctk.CTkLabel(self.sidebar, text="🧠 NeuralHealth", font=ctk.CTkFont(size=24, weight="bold")).pack(pady=30)
        
        self.lbl_temp = self.create_sensor_card("🌡️ Temp/Humid", "Wait...", "#3498db")
        self.lbl_noise = self.create_sensor_card("🔊 Noise Level", "Wait...", "#3498db")
        self.lbl_posture = self.create_sensor_card("🪑 Posture", "Active", "gray")
        self.lbl_emo = self.create_sensor_card("😐 Mood", "Neutral", "gray")
        
        ctk.CTkLabel(self.sidebar, text="ESP32 Status:", font=("Arial", 10)).pack(side="bottom", pady=5)
        self.lbl_conn = ctk.CTkLabel(self.sidebar, text="Connecting..." if self.brain.ser else "SIMULATION", font=("Arial", 10, "bold"), text_color="orange")
        self.lbl_conn.pack(side="bottom", pady=(0, 20))

        # Main Area
        self.main_frame = ctk.CTkFrame(self, corner_radius=10, fg_color="transparent")
        self.main_frame.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)

        self.video_frame = ctk.CTkLabel(self.main_frame, text="")
        self.video_frame.pack(fill="both", expand=True, pady=(0, 20))

        self.bot_frame = ctk.CTkFrame(self.main_frame, height=180, fg_color="#1a1a1a")
        self.bot_frame.pack(fill="x", side="bottom")

        ctk.CTkLabel(self.bot_frame, text="TOTAL STRESS LEVEL", font=("Arial", 12)).place(x=30, y=25)
        self.lbl_stress_val = ctk.CTkLabel(self.bot_frame, text="0%", font=("Arial", 45, "bold"), text_color="#00ff00")
        self.lbl_stress_val.place(x=30, y=55)

        self.lbl_advice = ctk.CTkLabel(self.bot_frame, text="System starting...", font=("Arial", 18), text_color="white", wraplength=550, justify="left")
        self.lbl_advice.place(x=220, y=50)

        self.update_gui()

    def create_sensor_card(self, title, val, color):
        frame = ctk.CTkFrame(self.sidebar, fg_color="#2b2b2b")
        frame.pack(pady=10, padx=15, fill="x")
        ctk.CTkLabel(frame, text=title, font=("Arial", 12, "bold"), text_color="gray").pack(anchor="w", padx=10, pady=(5,0))
        lbl = ctk.CTkLabel(frame, text=val, font=("Arial", 16), text_color=color)
        lbl.pack(anchor="w", padx=10, pady=(0,5))
        return lbl

    def calculate_stress_and_advice(self, pos, emo, noise, temp, hum):
        # (This logic stays exactly the same as you had it)
        score = 0
        msgs = []
        color = "#00ff00" 

        if "Slouch" in pos: 
            score += 35
            msgs.append("Sit up straight!")
        elif "Lean" in pos: 
            score += 30
            msgs.append("Correct your spine alignment.")

        if emo in ['Angry', 'Fear', 'Sad', 'Disgust']:
            score += 25
            msgs.append("High tension detected. Take deep breaths.")
        
        if noise > 65: 
            score += 15
            msgs.append("Noise levels are too high.")
        
        if temp > 30: 
            score += 10
            msgs.append(f"It's too hot ({temp}°C).")
        elif temp < 18 and temp > 10:
            score += 10
            msgs.append(f"It's chilly ({temp}°C).")
            
        score = min(score, 100)

        if score > 75:
            color = "#ff0000"
            main_msg = "⚠️ CRITICAL STRESS"
        elif score > 40:
            color = "#ffff00"
            main_msg = "⚠️ MODERATE STRESS"
        else:
            main_msg = "✅ Optimal State"
            if len(msgs) == 0: msgs.append("Working conditions look good.")

        final_advice = f"{main_msg}: {msgs[0]}" if msgs else main_msg
        return score, final_advice, color

    def update_gui(self):
        """
        THIS IS THE FIXED LOOP. 
        It does ZERO heavy math. It just reads variables.
        """
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            
            # 1. SEND FRAME TO BRAIN (Non-Blocking)
            # Only send if the brain is ready for a new one
            if self.brain.frame_to_process is None:
                self.brain.frame_to_process = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # 2. READ RESULTS (Instant)
            # We don't calculate them here. We just ask "What did you find?"
            pos_txt = self.brain.latest_pos
            emo_txt = self.brain.latest_emo
            noise = self.brain.latest_noise
            temp = self.brain.latest_temp
            hum = self.brain.latest_hum

            # 3. UPDATE UI LABELS
            if self.brain.ser:
                self.lbl_conn.configure(text="CONNECTED", text_color="#00ff00")
            
            self.lbl_temp.configure(text=f"{temp}°C | {hum:.0f}%")
            self.lbl_noise.configure(text=f"{noise:.1f} dB", text_color="red" if noise > 65 else "#3498db")
            self.lbl_posture.configure(text=pos_txt, text_color="red" if "Slouch" in pos_txt else "green")
            self.lbl_emo.configure(text=emo_txt, text_color="green" if emo_txt in ['Happy', 'Neutral'] else "red")

            # 4. DRAW OVERLAYS (Visuals only)
            disp_frame = frame.copy()
            p_color = (0, 255, 0)
            if "Slouch" in pos_txt: p_color = (255, 0, 0)
            
            cv2.putText(disp_frame, f"STATUS: {pos_txt}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, p_color, 2)
            cv2.putText(disp_frame, f"MOOD: {emo_txt}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # 5. CALCULATE SCORE
            score, advice, s_color = self.calculate_stress_and_advice(pos_txt, emo_txt, noise, temp, hum)
            self.lbl_stress_val.configure(text=f"{score}%", text_color=s_color)
            self.lbl_advice.configure(text=advice)
            
            # 6. SHOW IMAGE
            img = Image.fromarray(cv2.cvtColor(disp_frame, cv2.COLOR_BGR2RGB))
            imgtk = ctk.CTkImage(light_image=img, dark_image=img, size=(780, 480))
            self.video_frame.configure(image=imgtk)

        # Run this FAST (10ms) because it's lightweight now!
        self.after(10, self.update_gui)

if __name__ == "__main__":
    app = StressApp()
    app.mainloop()