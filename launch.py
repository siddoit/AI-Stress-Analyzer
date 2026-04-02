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
import ctypes
from hsemotion.facial_emotions import HSEmotionRecognizer
import requests

try:
    import sounddevice as sd
    HAS_LAPTOP_MIC = True
except ImportError:
    HAS_LAPTOP_MIC = False
    print("⚠️ 'sounddevice' library not found. Laptop Mic feature disabled.")

BAUD_RATE = 115200      
HISTORY_LEN = 20        
SKIP_FRAMES = 3         

# Added a Light/Eye Strain exercise
EXERCISES = {
    "Slouching": ("⬇Thoracic Extension", "Sit tall, arms behind head, arch back over chair."),
    "Leaning": ("Spine Re-alignment", "Stand up, reach high, bend side to side."),
    "High Stress": ("Box Breathing", "Inhale 4s, Hold 4s, Exhale 4s, Hold 4s."),
    "Noise": ("Auditory Break", "Wear noise-canceling headphones or step out."),
    "Heat": ("Hydration & Cool Down", "Drink water and check ventilation."),
    "Dim Light": ("Illuminate", "Eye strain detected. Turn on a lamp or open a window.")
}

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
_original_load = torch.load
def fixed_load(*args, **kwargs):
    if 'weights_only' not in kwargs: kwargs['weights_only'] = False
    return _original_load(*args, **kwargs)
torch.load = fixed_load

class StressBrain:
    def __init__(self):
        self.latest_pos = "Good Posture"
        self.latest_emo = "Neutral"
        self.latest_noise = 40.0
        self.latest_temp = 25.0
        self.latest_hum = 50.0
        self.latest_light = 300.0 # NEW: Light baseline
        
        self.use_laptop_mic = False 
        self.trigger_buzzer = False 
        
        self.stress_history = deque(maxlen=50) 
        self.care_prediction = "Stable"
        
        self.frame_to_process = None
        self.running = True
        
        self.ser = None
        self.connect_hardware()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        try:
            self.emo_model = HSEmotionRecognizer(model_name='enet_b0_8_best_vgaf', device=self.device)
        except Exception as e:
            self.emo_model = None

        try:
            self.posture_model = tf.keras.models.load_model('Models/my_posture_model.h5')
            self.pos_classes = ['Good', 'Slouching', 'Leaning']
        except:
            self.posture_model = None

        self.mp_face = mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)
        self.mp_pose = mp.solutions.pose.Pose(model_complexity=0, min_detection_confidence=0.5)
        self.pos_hist = deque(maxlen=10) 

    def connect_hardware(self):
        ports = list(serial.tools.list_ports.comports())
        target_port = None
        for p in ports:
            if "USB" in p.description or "CP210" in p.description:
                target_port = p.device
                break
        if target_port:
            try:
                # Stop Python from sending the DTR/RTS reboot signal
                self.ser = serial.Serial()
                self.ser.port = target_port
                self.ser.baudrate = BAUD_RATE
                self.ser.timeout = 0.1
                self.ser.setDTR(False) 
                self.ser.setRTS(False)
                self.ser.open()
                print(f"✅ Hardware Connected without Rebooting: {target_port}")
            except: 
                print("⚠️ Hardware Found but Busy.")

    def start(self):
        t = threading.Thread(target=self._worker_loop)
        t.daemon = True
        t.start()

    def _get_laptop_noise_level(self):
        if not HAS_LAPTOP_MIC: return 40.0
        try:
            recording = sd.rec(int(0.1 * 44100), samplerate=44100, channels=1, dtype='float32')
            sd.wait()
            rms = np.sqrt(np.mean(recording**2))
            if rms > 0:
                return max(30.0, 20 * np.log10(rms) + 90)
        except: pass
        return 40.0

    def _worker_loop(self):
        frame_count = 0
        while self.running:
            if self.use_laptop_mic and frame_count % 10 == 0:
                self.latest_noise = self._get_laptop_noise_level()
            
            if self.ser:
                if self.trigger_buzzer:
                    try: self.ser.write(b'BUZZ\n')
                    except: pass
                    self.trigger_buzzer = False

                if self.ser.in_waiting > 0:
                    try:
                        line = self.ser.readline().decode('utf-8', errors='ignore').strip()
                        if "Sound:" in line:
                            parts = line.split('|')
                            if not self.use_laptop_mic:
                                self.latest_noise = float(parts[0].split(':')[1].replace('dB', ''))
                            self.latest_temp = float(parts[1].split(':')[1].replace('C', ''))
                            if len(parts) > 2:
                                self.latest_hum = float(parts[2].split(':')[1].replace('%', ''))
                            if len(parts) > 3: # NEW: Parsing Light
                                self.latest_light = float(parts[3].split(':')[1].replace('lx', ''))
                    except: pass

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
        H, W, _ = frame.shape
        person_present = False
        
        # 1. PRESENCE & PROXIMITY CHECK (FACE)
        f_res = self.mp_face.process(frame)
        if f_res.detections:
            d = f_res.detections[0]
            bb = d.location_data.relative_bounding_box
            x, y = int(bb.xmin * W), int(bb.ymin * H)
            w, h = int(bb.width * W), int(bb.height * H)
            
            # Require the face width to be at least 8% of the camera width.
            # If it's smaller, they are standing up or across the room.
            if w > (W * 0.08) and x >= 0 and y >= 0 and (x+w) <= W and (y+h) <= H:
                person_present = True
                if self.emo_model:
                    f_img = frame[y:y+h, x:x+w]
                    self.latest_emo, _ = self.emo_model.predict_emotions(f_img, logits=False)
            else:
                self.latest_emo = "Away"
        else:
            self.latest_emo = "Away"

        # 2. POSTURE CHECK (ONLY IF SEATED AT DESK)
        if person_present:
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
            else:
                self.latest_pos = "Calculating..."
        else:
            self.latest_pos = "Away"
            self.pos_hist.clear() # Dump the old history so it doesn't drag


class StressApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.last_log_time = 0  
        self.last_interrupt_time = 0
        self.critical_start_time = 0  # NEW: Tracks how long you've been stressed
        
        self.telegram_token = "8016038592:AAFGAAX6MgPUjOvu6LQfGwfwtMrTmNcv9lk"
        self.telegram_chat_id = "8758118564"
        
        self.brain = StressBrain()
        self.brain.start()

        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        self.title("Stress Analysis & Care Detection")
        self.geometry("1300x850")
        ctk.set_appearance_mode("Dark")
        ctk.set_default_color_theme("blue")

        self.grid_columnconfigure(0, weight=0, minsize=220)
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure(2, weight=0, minsize=300)
        self.grid_rowconfigure(0, weight=1)

        self._init_ui()
        self.update_gui()

    def _init_ui(self):
        self.sidebar = ctk.CTkFrame(self, width=220, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        
        ctk.CTkLabel(self.sidebar, text="SYSTEM CONTROLS", font=("Roboto Medium", 16)).pack(pady=(30,10))
        
        self.mic_switch = ctk.CTkSwitch(self.sidebar, text="Use Laptop Mic", command=self._toggle_mic)
        self.mic_switch.pack(pady=10, padx=20, anchor="w")
        if not HAS_LAPTOP_MIC:
            self.mic_switch.configure(state="disabled", text="Laptop Mic (N/A)")

        ctk.CTkLabel(self.sidebar, text="ENVIRONMENTAL", font=("Roboto Medium", 16)).pack(pady=(30,10))
        
        self.card_temp = self._create_sensor_card("🌡️ TEMPERATURE", "25°C", "#FF9F1C") 
        self.card_hum = self._create_sensor_card("💧 HUMIDITY", "50%", "#4CC9F0")     
        self.card_noise = self._create_sensor_card("🔊 NOISE LEVEL", "40 dB", "#F72585") 
        self.card_light = self._create_sensor_card("☀️ LIGHT LEVEL", "300 lx", "#FFD166") # NEW: Light UI Card

        self.lbl_conn = ctk.CTkLabel(self.sidebar, text="● ESP32 SEARCHING...", text_color="gray", font=("Arial", 10))
        self.lbl_conn.pack(side="bottom", pady=20)

        self.main_area = ctk.CTkFrame(self, fg_color="transparent")
        self.main_area.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        
        self.vid_container = ctk.CTkFrame(self.main_area, fg_color="#000000", corner_radius=15)
        self.vid_container.pack(fill="both", expand=True)
        
        self.vid_lbl = ctk.CTkLabel(self.vid_container, text=" ", text_color="gray")
        self.vid_lbl.pack(fill="both", expand=True, padx=2, pady=2)

        self.care_panel = ctk.CTkFrame(self, width=300, corner_radius=0, fg_color="#1A1A1A")
        self.care_panel.grid(row=0, column=2, sticky="nsew")
        
        ctk.CTkLabel(self.care_panel, text="CARE DASHBOARD", font=("Roboto", 24, "bold")).pack(pady=(30, 20))

        self.stress_card = ctk.CTkFrame(self.care_panel, fg_color="#2B2B2B", corner_radius=10)
        self.stress_card.pack(fill="x", padx=20, pady=10)
        ctk.CTkLabel(self.stress_card, text="NET STRESS LOAD", font=("Arial", 12, "bold"), text_color="gray").pack(pady=(10,0))
        self.lbl_score = ctk.CTkLabel(self.stress_card, text="0%", font=("Impact", 60), text_color="#00FF00")
        self.lbl_score.pack(pady=(0,10))

        ctk.CTkLabel(self.care_panel, text="PRIMARY FACTORS", font=("Roboto", 14, "bold"), text_color="gray").pack(pady=(20, 5), anchor="w", padx=20)
        self.card_pos = self._create_status_row("Posture", "Active", "white")
        self.card_emo = self._create_status_row("Emotion", "Neutral", "white")

        ctk.CTkLabel(self.care_panel, text="CARE PREDICTION", font=("Roboto", 14, "bold"), text_color="gray").pack(pady=(30, 5), anchor="w", padx=20)
        self.lbl_pred = ctk.CTkLabel(self.care_panel, text="Stable State", font=("Arial", 18, "bold"), text_color="#00FF00")
        self.lbl_pred.pack(anchor="w", padx=20)
        
        self.rec_box = ctk.CTkFrame(self.care_panel, fg_color="#333333")
        self.rec_box.pack(fill="x", padx=20, pady=10)
        self.lbl_action_title = ctk.CTkLabel(self.rec_box, text="Monitoring...", font=("Arial", 14, "bold"), text_color="#FF9F1C")
        self.lbl_action_title.pack(pady=(10,0))
        self.lbl_action_desc = ctk.CTkLabel(self.rec_box, text="System is establishing baseline.", font=("Arial", 12), wraplength=220)
        self.lbl_action_desc.pack(pady=(5,10))

        self.txt_log = ctk.CTkTextbox(self.care_panel, height=150, fg_color="#111", text_color="#0f0", font=("Consolas", 10))
        self.txt_log.pack(fill="x", padx=20, pady=20, side="bottom")

    def _create_sensor_card(self, title, val, color):
        f = ctk.CTkFrame(self.sidebar, fg_color="#212121")
        f.pack(pady=5, padx=15, fill="x")
        ctk.CTkLabel(f, text=title, font=("Arial", 10, "bold"), text_color="gray").pack(anchor="w", padx=10, pady=(5,0))
        lbl = ctk.CTkLabel(f, text=val, font=("Arial", 16, "bold"), text_color=color)
        lbl.pack(anchor="w", padx=10, pady=(0,5))
        return lbl

    def _create_status_row(self, title, val, color):
        f = ctk.CTkFrame(self.care_panel, fg_color="transparent")
        f.pack(fill="x", padx=20, pady=2)
        ctk.CTkLabel(f, text=title, font=("Arial", 14), text_color="gray").pack(side="left")
        lbl = ctk.CTkLabel(f, text=val, font=("Arial", 14, "bold"), text_color=color)
        lbl.pack(side="right")
        return lbl

    def _toggle_mic(self):
        self.brain.use_laptop_mic = bool(self.mic_switch.get())
        src = "LAPTOP" if self.brain.use_laptop_mic else "ESP32"
        self.log_event(f"Audio Source Switched to: {src}")

    def fire_windows_interrupt(self):
        ctypes.windll.user32.MessageBoxW(0, "CRITICAL STRESS DETECTED.\nStep away immediately.", "CARE SYSTEM OVERRIDE", 0x1000 | 0x30)

    def log_event(self, message):
        timestamp = time.strftime("%H:%M:%S")
        self.txt_log.insert("end", f"[{timestamp}] {message}\n")
        self.txt_log.see("end") 

    def calculate_logic(self):
        pos = self.brain.latest_pos
        emo = self.brain.latest_emo
        noise = self.brain.latest_noise
        temp = self.brain.latest_temp
        light = self.brain.latest_light 
        
        score = 0
        recs = []
        current_time = time.time()
        trigger_msg = None 

        # --- THE NEW BALANCED WEIGHTS ---
        # Posture: 35% | Emotion: 35% | Environment: 30%

        # 1. POSTURE & EMOTION (ONLY IF AT DESK)
        if pos != "Away" and emo != "Away":
            # Posture Check (Max 35)
            if "Slouch" in pos: 
                score += 35
                recs.append(EXERCISES["Slouching"])
                trigger_msg = "Posture: Slouching"
            elif "Lean" in pos: 
                score += 20
                recs.append(EXERCISES["Leaning"])
                if not trigger_msg: trigger_msg = "Posture: Leaning"

            # Emotion Check (Max 35)
            if emo in ['Angry', 'Fear', 'Sad']: 
                score += 35
                recs.append(EXERCISES["High Stress"])
                if not trigger_msg: trigger_msg = f"Mood: {emo}"
            elif emo == 'Disgust':
                score += 20
                if not trigger_msg: trigger_msg = "Mood: Disgust"
        else:
            # User is away. Stop adding visual stress.
            self.lbl_action_title.configure(text="Standby")
            self.lbl_action_desc.configure(text="Waiting for user to return to desk.")

        # 2. ENVIRONMENT (Max 30 - Always tracking so the room is ready)
        if noise > 75: 
            score += 15
            recs.append(EXERCISES["Noise"])
            if not trigger_msg: trigger_msg = "Env: Noise High"
        
        if temp > 30: 
            score += 10
            recs.append(EXERCISES["Heat"])

        if light < 40: 
            score += 5
            recs.append(EXERCISES["Dim Light"])
            if not trigger_msg: trigger_msg = "Env: Too Dark (Eye Strain)"
            
        score = min(score, 100)
        
        # --- TREND ANALYSIS & TRIGGERS ---
        self.brain.stress_history.append(score)
        avg_stress = sum(self.brain.stress_history) / len(self.brain.stress_history) if self.brain.stress_history else 0
        
        pred_text = "Optimal State"
        pred_col = "#00FF00" 

        if pos == "Away":
            pred_text = "User Away"
            pred_col = "gray"
            self.critical_start_time = 0 # Reset critical timer if they stand up
        else:
            # We lowered the threshold triggers slightly to match the new math
            if avg_stress > 30:
                pred_text = "Burnout Risk"
                pred_col = "#FF9F1C" 
            
            if avg_stress > 50: 
                pred_text = "CRITICAL LOAD"
                pred_col = "#FF0000" 
                trigger_msg = "ALERT: Burnout Threshold!"
                
                # CRITICAL EVENT EXECUTION
                if self.critical_start_time == 0:
                    self.critical_start_time = current_time
                elif current_time - self.critical_start_time >= 5.0:
                    if current_time - self.last_interrupt_time > 60:
                        
                        threading.Thread(target=self.fire_windows_interrupt, daemon=True).start()
                        
                        self.send_telegram_alert(
                            score=score, level="High", pos=self.brain.latest_pos,
                            emo=self.brain.latest_emo, noise=self.brain.latest_noise
                        )
                        
                        def delayed_buzzer():
                            time.sleep(1.5)
                            self.brain.trigger_buzzer = True
                            
                        threading.Thread(target=delayed_buzzer, daemon=True).start()
                        self.last_interrupt_time = current_time
            else:
                self.critical_start_time = 0

        if trigger_msg and (current_time - self.last_log_time > 5) and pos != "Away":
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
            
            if self.brain.ser: self.lbl_conn.configure(text="● ESP32 CONNECTED", text_color="#00FF00")
            else: self.lbl_conn.configure(text="● SIMULATION MODE", text_color="orange")

            self.card_temp.configure(text=f"{self.brain.latest_temp}°C")
            self.card_hum.configure(text=f"{self.brain.latest_hum}%")
            self.card_noise.configure(text=f"{self.brain.latest_noise:.1f} dB")
            self.card_light.configure(text=f"{self.brain.latest_light:.0f} lx") # NEW: Update light UI
            
            self.card_pos.configure(text=self.brain.latest_pos, text_color="red" if "Slouch" in self.brain.latest_pos else "white")
            self.card_emo.configure(text=self.brain.latest_emo, text_color="red" if self.brain.latest_emo in ['Angry', 'Fear'] else "white")
            
            self.lbl_score.configure(text=f"{score}%", text_color=pred_col)
            self.lbl_pred.configure(text=pred_text, text_color=pred_col)

            if recs:
                self.lbl_action_title.configure(text=recs[0][0])
                self.lbl_action_desc.configure(text=recs[0][1])
            else:
                self.lbl_action_title.configure(text="Condition Nominal")
                self.lbl_action_desc.configure(text="Continue current workflow.")

            cv2.putText(frame, f"STRESS: {score}%", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255) if score>50 else (0,255,0), 3)
            
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            w = self.vid_container.winfo_width()
            h = self.vid_container.winfo_height()
            if w > 10 and h > 10:
                imgtk = ctk.CTkImage(light_image=img, dark_image=img, size=(w, h))
                self.vid_lbl.configure(image=imgtk)
        
        self.after(20, self.update_gui)
        
    def send_telegram_alert(self, score, level, pos, emo, noise):
        if not self.telegram_token or not self.telegram_chat_id:
            return

        url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
        
        # Formatting your exact message structure
        msg = (
            f"🚨 Telegram notification 🚨\n\n"
            f"Stress level : {level}\n"
            f"Stress score : {int(score)}%\n"
            f"Posture : {pos}\n"
            f"Emotion : {emo}\n"
            f"Noiselevel : {noise:.0f} db"
        )
        
        payload = {"chat_id": self.telegram_chat_id, "text": msg}
        
        # We thread this so the UI doesn't lag
        def post_req():
            try:
                requests.post(url, json=payload, timeout=5)
                self.log_event("Telegram alert delivered.")
            except Exception:
                self.log_event("Network error: Telegram alert failed.")
                
        threading.Thread(target=post_req, daemon=True).start()

if __name__ == "__main__":
    app = StressApp()
    app.mainloop()