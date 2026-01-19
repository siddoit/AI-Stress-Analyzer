import customtkinter as ctk
import cv2
import torch
import numpy as np
import mediapipe as mp
import tensorflow as tf
import os
import shutil
import statistics
import serial
import serial.tools.list_ports
from pathlib import Path
from PIL import Image, ImageTk
from collections import deque
import threading
import time
from hsemotion.facial_emotions import HSEmotionRecognizer

# --- CONFIGURATION ---
BAUD_RATE = 115200      
HISTORY_LEN = 10        
SKIP_FRAMES = 5         # OPTIMIZATION: Only run AI every 5 frames

# --- EXERCISE DATABASE ---
EXERCISES = {
    "Slouching": ("⬇️ Thoracic Extension", "Sit tall, arms behind head, arch back over chair."),
    "Leaning": ("⚖️ Spine Re-alignment", "Stand up, reach high, bend side to side."),
    "High Stress": ("🌬️ Box Breathing", "Inhale 4s, Hold 4s, Exhale 4s, Hold 4s."),
    "Noise": ("🎧 Auditory Break", "Wear noise-canceling headphones or step out."),
    "Heat": ("💧 Hydration & Cool Down", "Drink water and check ventilation.")
}

# System Fixes for Windows/TF
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
_original_load = torch.load
def fixed_load(*args, **kwargs):
    if 'weights_only' not in kwargs: kwargs['weights_only'] = False
    return _original_load(*args, **kwargs)
torch.load = fixed_load

# ==========================================
# 🧠 BACKEND (THE BRAIN)
# ==========================================
class StressBrain:
    def __init__(self):
        # Shared Data
        self.latest_pos = "Good Posture"
        self.latest_emo = "Neutral"
        self.latest_noise = 40.0
        self.latest_temp = 25.0
        self.latest_hum = 50.0
        
        # Trend Analysis
        self.stress_history = deque(maxlen=50) 
        self.care_prediction = "Stable"
        
        self.frame_to_process = None
        self.running = True
        
        # --- HARDWARE CONNECTION ---
        self.ser = None
        self.connect_hardware()

        # --- AI LOAD ---
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"🚀 AI Running on: {self.device.upper()}")

        try:
            self.emo_model = HSEmotionRecognizer(model_name='enet_b0_8_best_vgaf', device=self.device)
        except Exception as e:
            print(f"⚠️ Emotion Model Failed: {e}")
            self.emo_model = None

        try:
            self.posture_model = tf.keras.models.load_model('Models/my_posture_model.h5')
            self.pos_classes = ['Good', 'Slouching', 'Leaning']
        except:
            print("⚠️ Posture Model Failed (Check path)")
            self.posture_model = None

        self.mp_face = mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)
        self.mp_pose = mp.solutions.pose.Pose(model_complexity=0, min_detection_confidence=0.5)
        self.pos_hist = deque(maxlen=5)

    def connect_hardware(self):
        ports = list(serial.tools.list_ports.comports())
        target_port = None
        for p in ports:
            if "USB" in p.description or "CP210" in p.description:
                target_port = p.device
                break
        
        if target_port:
            try:
                self.ser = serial.Serial(target_port, BAUD_RATE, timeout=0.1)
                print(f"✅ Hardware Connected: {target_port}")
            except:
                print("⚠️ Hardware Found but Busy.")
        else:
            print("⚠️ No Hardware Found (Simulation Mode)")

    def start(self):
        t = threading.Thread(target=self._worker_loop)
        t.daemon = True
        t.start()

    def _worker_loop(self):
        frame_count = 0
        while self.running:
            # 1. READ HARDWARE
            if self.ser and self.ser.in_waiting > 0:
                try:
                    line = self.ser.readline().decode('utf-8', errors='ignore').strip()
                    if "Sound:" in line:
                        parts = line.split('|')
                        self.latest_noise = float(parts[0].split(':')[1].replace('dB', ''))
                        self.latest_temp = float(parts[1].split(':')[1].replace('C', ''))
                        if len(parts) > 2:
                            self.latest_hum = float(parts[2].split(':')[1].replace('%', ''))
                except: pass

            # 2. AI PROCESS (Throttled)
            if self.frame_to_process is not None:
                frame_count += 1
                if frame_count % SKIP_FRAMES == 0:
                    curr_frame = self.frame_to_process.copy()
                    self.frame_to_process = None 
                    self._run_inference(curr_frame)
                else:
                    self.frame_to_process = None 

            time.sleep(0.01)

    def _run_inference(self, frame):
        # Posture
        results = self.mp_pose.process(frame)
        if results.pose_landmarks and self.posture_model:
            lm = []
            for i in range(17):
                p = results.pose_landmarks.landmark[i]
                lm.extend([p.x, p.y, p.z, p.visibility])
            try:
                pred = self.posture_model.predict(np.array(lm).reshape(1,-1), verbose=0)
                idx = np.argmax(pred)
                self.pos_hist.append(idx)
                self.latest_pos = self.pos_classes[statistics.mode(self.pos_hist)]
            except: pass

        # Emotion
        f_res = self.mp_face.process(frame)
        if f_res.detections and self.emo_model:
            d = f_res.detections[0]
            bb = d.location_data.relative_bounding_box
            H, W, _ = frame.shape
            x, y, w, h = int(bb.xmin*W), int(bb.ymin*H), int(bb.width*W), int(bb.height*H)
            if w > 20 and x >= 0 and y >= 0 and (x+w) <= W and (y+h) <= H:
                f_img = frame[y:y+h, x:x+w]
                self.latest_emo, _ = self.emo_model.predict_emotions(f_img, logits=False)

# ==========================================
# 🖥️ FRONTEND (THE FIX)
# ==========================================
class StressApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.last_log_time = 0  # <--- FIXED: Added this missing variable
        
        self.brain = StressBrain()
        self.brain.start()

        self.cap = cv2.VideoCapture(0)
        
        # UI Setup
        self.title("NeuroErgo: Care Prediction System")
        self.geometry("1280x720")
        ctk.set_appearance_mode("Dark")
        ctk.set_default_color_theme("dark-blue")

        self.grid_columnconfigure(1, weight=3)
        self.grid_columnconfigure(2, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self._setup_sidebar()
        self._setup_main_area()
        self._setup_care_portal()
        
        self.update_gui()

    # --- FIXED LOGIC FOR LOGGING ---
    def log_event(self, message):
        """Adds a timestamped line to the session health box"""
        timestamp = time.strftime("%H:%M:%S")
        self.txt_log.insert("end", f"[{timestamp}] {message}\n")
        self.txt_log.see("end") 

    def _setup_sidebar(self):
        self.sidebar = ctk.CTkFrame(self, width=200, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        
        ctk.CTkLabel(self.sidebar, text="⚡ SYSTEM STATUS", font=("Impact", 20)).pack(pady=20)
        
        self.lbl_conn = ctk.CTkLabel(self.sidebar, text="SEARCHING...", text_color="orange", font=("Arial", 10, "bold"))
        self.lbl_conn.pack(pady=(0, 20))

        self.card_temp = self._create_card("TEMP / HUMID", "25°C | 50%", "#00BFFF")
        self.card_noise = self._create_card("NOISE LEVEL", "40 dB", "#00BFFF")
        self.card_pos = self._create_card("POSTURE", "Analyzing...", "gray")
        self.card_emo = self._create_card("EMOTION", "Analyzing...", "gray")

    def _create_card(self, title, val, color):
        f = ctk.CTkFrame(self.sidebar, fg_color="#1F1F1F")
        f.pack(pady=8, padx=10, fill="x")
        ctk.CTkLabel(f, text=title, font=("Arial", 10, "bold"), text_color="gray").pack(anchor="w", padx=10, pady=(5,0))
        lbl = ctk.CTkLabel(f, text=val, font=("Arial", 14, "bold"), text_color=color)
        lbl.pack(anchor="w", padx=10, pady=(0,5))
        return lbl

    def _setup_main_area(self):
        self.main = ctk.CTkFrame(self, fg_color="transparent")
        self.main.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        
        self.vid_lbl = ctk.CTkLabel(self.main, text="")
        self.vid_lbl.pack(fill="both", expand=True, pady=(0, 10))

        self.bot_bar = ctk.CTkFrame(self.main, height=100, fg_color="#111111")
        self.bot_bar.pack(fill="x")
        
        ctk.CTkLabel(self.bot_bar, text="LIVE STRESS INDEX", font=("Arial", 12)).place(x=20, y=10)
        self.lbl_score = ctk.CTkLabel(self.bot_bar, text="0%", font=("Impact", 40), text_color="green")
        self.lbl_score.place(x=20, y=35)
        
        self.lbl_pred = ctk.CTkLabel(self.bot_bar, text="PREDICTION: Stable", font=("Arial", 16, "bold"), text_color="gray")
        self.lbl_pred.place(x=150, y=45)

    def _setup_care_portal(self):
        self.care_frame = ctk.CTkFrame(self, width=250, corner_radius=0)
        self.care_frame.grid(row=0, column=2, sticky="nsew")
        
        ctk.CTkLabel(self.care_frame, text="🏥 CARE PORTAL", font=("Impact", 20)).pack(pady=20)
        
        self.rec_box = ctk.CTkFrame(self.care_frame, fg_color="#2B2B2B")
        self.rec_box.pack(pady=10, padx=10, fill="x")
        ctk.CTkLabel(self.rec_box, text="RECOMMENDED ACTION", font=("Arial", 11, "bold"), text_color="orange").pack(pady=5)
        self.lbl_action_title = ctk.CTkLabel(self.rec_box, text="None", font=("Arial", 14, "bold"))
        self.lbl_action_title.pack()
        self.lbl_action_desc = ctk.CTkLabel(self.rec_box, text="System is monitoring...", font=("Arial", 12), wraplength=200)
        self.lbl_action_desc.pack(pady=10)

        ctk.CTkLabel(self.care_frame, text="SESSION HEALTH", font=("Impact", 16)).pack(pady=(30,10))
        self.txt_log = ctk.CTkTextbox(self.care_frame, height=200)
        self.txt_log.pack(padx=10, fill="x")
        self.txt_log.insert("0.0", "System Initialization...\nMonitoring Started.\n")

    def calculate_logic(self):
        # 1. Get Data
        pos = self.brain.latest_pos
        emo = self.brain.latest_emo
        noise = self.brain.latest_noise
        temp = self.brain.latest_temp
        
        score = 0
        recs = []
        current_time = time.time()
        
        trigger_msg = None 

        # Posture Check
        if "Slouch" in pos: 
            score += 30
            recs.append(EXERCISES["Slouching"])
            trigger_msg = "Posture: Slouching detected"
        elif "Lean" in pos: 
            score += 25
            recs.append(EXERCISES["Leaning"])
            trigger_msg = "Posture: Leaning too much"
            
        # Emotion Check
        if emo in ['Angry', 'Fear', 'Sad']: 
            score += 25
            recs.append(EXERCISES["High Stress"])
            if not trigger_msg: trigger_msg = f"Mood: Detected {emo}"
            
        # Environment Check
        if noise > 70: 
            score += 15
            recs.append(EXERCISES["Noise"])
            if not trigger_msg: trigger_msg = "Environment: Noise levels high"
            
        if temp > 29: 
            score += 10
            recs.append(EXERCISES["Heat"])
            if not trigger_msg: trigger_msg = f"Environment: High Temp ({temp}°C)"
            
        score = min(score, 100)
        
        # --- CARE PREDICTION ---
        self.brain.stress_history.append(score)
        avg_stress = sum(self.brain.stress_history) / len(self.brain.stress_history) if self.brain.stress_history else 0
        
        pred_text = "Stable Flow"
        pred_col = "green"
        
        if avg_stress > 60:
            pred_text = "⚠️ BURNOUT RISK RISING"
            pred_col = "orange"
        if avg_stress > 80:
            pred_text = "🚨 CRITICAL: TAKE A BREAK"
            pred_col = "red"
            trigger_msg = "ALERT: Burnout Threshold Reached!"

        # --- LOGGING WITH COOLDOWN ---
        if trigger_msg and (current_time - self.last_log_time > 5):
            self.log_event(trigger_msg)
            self.last_log_time = current_time
            
        return score, recs, pred_text, pred_col

    def update_gui(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            
            if self.brain.frame_to_process is None:
                self.brain.frame_to_process = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
            score, recs, pred_text, pred_col = self.calculate_logic()
            
            if self.brain.ser: self.lbl_conn.configure(text="CONNECTED (ESP32)", text_color="#00FF00")
            else: self.lbl_conn.configure(text="SIMULATION MODE", text_color="orange")

            self.card_temp.configure(text=f"{self.brain.latest_temp}°C | {self.brain.latest_hum}%")
            self.card_noise.configure(text=f"{self.brain.latest_noise:.1f} dB")
            self.card_pos.configure(text=self.brain.latest_pos, text_color="red" if "Slouch" in self.brain.latest_pos else "green")
            self.card_emo.configure(text=self.brain.latest_emo)
            
            self.lbl_score.configure(text=f"{score}%", text_color=pred_col)
            self.lbl_pred.configure(text=pred_text, text_color=pred_col)

            if recs:
                top_rec = recs[0]
                self.lbl_action_title.configure(text=top_rec[0])
                self.lbl_action_desc.configure(text=top_rec[1])
            else:
                self.lbl_action_title.configure(text="All Good")
                self.lbl_action_desc.configure(text="Maintain current workflow.")

            cv2.putText(frame, f"NET STRESS: {score}%", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255) if score>50 else (0,255,0), 2)
            
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            imgtk = ctk.CTkImage(light_image=img, dark_image=img, size=(700, 450))
            self.vid_lbl.configure(image=imgtk)
        
        self.after(20, self.update_gui)

if __name__ == "__main__":
    app = StressApp()
    app.mainloop()