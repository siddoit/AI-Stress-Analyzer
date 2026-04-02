import os
import warnings
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Suppress TensorFlow and MediaPipe C++ logging BEFORE importing them
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['GLOG_minloglevel'] = '2'
warnings.filterwarnings('ignore')

import cv2
import torch
import numpy as np
import mediapipe as mp
import tensorflow as tf
import statistics
import serial
import serial.tools.list_ports
import threading
import time
import ctypes
import requests
from collections import deque
from hsemotion.facial_emotions import HSEmotionRecognizer

try:
    import sounddevice as sd
    HAS_LAPTOP_MIC = True
except ImportError:
    HAS_LAPTOP_MIC = False
    print("Warning: 'sounddevice' library not found. Laptop Mic feature disabled.")

BAUD_RATE = 115200      
SKIP_FRAMES = 3        

EXERCISES = {
    "Slouching": ("⬇Thoracic Extension", "Sit tall, arms behind head, arch back over chair."),
    "Leaning": ("Spine Re-alignment", "Stand up, reach high, bend side to side."),
    "High Stress": ("Box Breathing", "Inhale 4s, Hold 4s, Exhale 4s, Hold 4s."),
    "Noise": ("Auditory Break", "Wear noise-canceling headphones or step out."),
    "Heat": ("Hydration & Cool Down", "Drink water and check ventilation."),
    "Dim Light": ("Illuminate", "Eye strain detected. Turn on a lamp or open a window.")
}

_original_load = torch.load
def fixed_load(*args, **kwargs):
    if 'weights_only' not in kwargs: kwargs['weights_only'] = False
    return _original_load(*args, **kwargs)
torch.load = fixed_load

class StressEngine:
    def __init__(self):
        # State variables
        self.latest_pos = "Good Posture"
        self.latest_emo = "Neutral"
        self.latest_noise = 40.0
        self.latest_temp = 25.0
        self.latest_hum = 50.0
        self.latest_light = 300.0 
        
        # Threading and running state
        self.running = False
        self.latest_camera_frame = None
        self.use_laptop_mic = False 
        self.trigger_buzzer = False 
        
        # Analysis states
        self.stress_history = deque(maxlen=50) 
        self.care_prediction = "Stable"
        self.current_score = 0
        self.current_recs = []
        self.pred_color = "#00FF00"
        self.latest_frame_jpeg = None
        self.event_logs = []
        
        self.last_log_time = 0  
        self.last_interrupt_time = 0
        self.critical_start_time = 0  
        
        self.telegram_token = os.getenv("TELEGRAM_BOT_TOKEN")
        self.telegram_chat_id = os.getenv("TELEGRAM_CHAT_ID")

        # Hardware
        self.ser = None
        self.hardware_status = "SEARCHING"
        self.connect_hardware()

        # Models
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

        # Camera Setup (DirectShow to kill latency, single buffer)
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def connect_hardware(self):
        ports = list(serial.tools.list_ports.comports())
        target_port = None
        for p in ports:
            if "USB" in p.description or "CP210" in p.description:
                target_port = p.device
                break
        if target_port:
            try:
                self.ser = serial.Serial()
                self.ser.port = target_port
                self.ser.baudrate = BAUD_RATE
                self.ser.timeout = 0.1
                self.ser.setDTR(False) 
                self.ser.setRTS(False)
                self.ser.open()
                self.hardware_status = "CONNECTED"
            except: 
                self.hardware_status = "BUSY"
        else:
            self.hardware_status = "SIMULATION"

    def log_event(self, message):
        timestamp = time.strftime("%H:%M:%S")
        self.event_logs.append(f"[{timestamp}] {message}")
        if len(self.event_logs) > 20:
            self.event_logs.pop(0)

    def start(self):
        self.running = True
        # Start both the camera grabber and the main processing loop AFTER running is True
        threading.Thread(target=self._read_camera_frames, daemon=True).start()
        threading.Thread(target=self._main_loop, daemon=True).start()

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

    def fire_windows_interrupt(self):
        ctypes.windll.user32.MessageBoxW(0, "CRITICAL STRESS DETECTED.\nStep away immediately.", "CARE SYSTEM OVERRIDE", 0x1000 | 0x30)

    def send_telegram_alert(self, score, level, pos, emo, noise):
        if not self.telegram_token or not self.telegram_chat_id: return
        url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
        msg = (
            f"🚨 Alert Telegram notification Alert 🚨\n\n"
            f"Stress level : {level}\n"
            f"Stress score : {int(score)}%\n"
            f"Posture : {pos}\n"
            f"Emotion : {emo}\n"
            f"Noiselevel : {noise:.0f} db"
        )
        payload = {"chat_id": self.telegram_chat_id, "text": msg}
        def post_req():
            try:
                requests.post(url, json=payload, timeout=5)
                self.log_event("Telegram alert delivered.")
            except Exception:
                self.log_event("Network error: Telegram alert failed.")
        threading.Thread(target=post_req, daemon=True).start()

    def calculate_logic(self):
        score = 0
        recs = []
        current_time = time.time()
        trigger_msg = None 

        if self.latest_pos != "Away" and self.latest_emo != "Away":
            if "Slouch" in self.latest_pos: 
                score += 35
                recs.append(EXERCISES["Slouching"])
                trigger_msg = "Posture: Slouching"
            elif "Lean" in self.latest_pos: 
                score += 20
                recs.append(EXERCISES["Leaning"])
                if not trigger_msg: trigger_msg = "Posture: Leaning"

            # Exactly matching HSEmotion's real outputs
            if self.latest_emo in ['Anger', 'Fear', 'Sadness']: 
                score += 35
                recs.append(EXERCISES["High Stress"])
                if not trigger_msg: trigger_msg = f"Mood: {self.latest_emo}"
            elif self.latest_emo == 'Disgust':
                score += 20
                if not trigger_msg: trigger_msg = "Mood: Disgust"
        
        if self.latest_noise > 75: 
            score += 15
            recs.append(EXERCISES["Noise"])
            if not trigger_msg: trigger_msg = "Env: Noise High"
        
        if self.latest_temp > 30: 
            score += 10
            recs.append(EXERCISES["Heat"])

        if self.latest_light < 40: 
            score += 5
            recs.append(EXERCISES["Dim Light"])
            if not trigger_msg: trigger_msg = "Env: Too Dark (Eye Strain)"
            
        score = min(score, 100)
        self.stress_history.append(score)
        avg_stress = sum(self.stress_history) / len(self.stress_history) if self.stress_history else 0
        
        pred_text = "Optimal State"
        pred_col = "#00FF00" 

        if self.latest_pos == "Away":
            pred_text = "User Away"
            pred_col = "gray"
            self.critical_start_time = 0 
        else:
            if avg_stress > 30:
                pred_text = "Burnout Risk"
                pred_col = "#FF9F1C" 
            
            if avg_stress > 50: 
                pred_text = "CRITICAL LOAD"
                pred_col = "#FF0000" 
                trigger_msg = "ALERT: Burnout Threshold!"
                
                if self.critical_start_time == 0:
                    self.critical_start_time = current_time
                elif current_time - self.critical_start_time >= 5.0:
                    if current_time - self.last_interrupt_time > 60:
                        threading.Thread(target=self.fire_windows_interrupt, daemon=True).start()
                        self.send_telegram_alert(score=score, level="High", pos=self.latest_pos, emo=self.latest_emo, noise=self.latest_noise)
                        
                        def delayed_buzzer():
                            time.sleep(1.5)
                            self.trigger_buzzer = True
                        threading.Thread(target=delayed_buzzer, daemon=True).start()
                        self.last_interrupt_time = current_time
            else:
                self.critical_start_time = 0

        if trigger_msg and (current_time - self.last_log_time > 5) and self.latest_pos != "Away":
            self.log_event(trigger_msg)
            self.last_log_time = current_time
            
        self.current_score = score
        self.current_recs = recs
        self.care_prediction = pred_text
        self.pred_color = pred_col

    def _main_loop(self):
        frame_count = 0
        while self.running:
            # Check laptop mic if enabled
            if self.use_laptop_mic and frame_count % 10 == 0:
                self.latest_noise = self._get_laptop_noise_level()
            
            # Process hardware serial data
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
                            if len(parts) > 3: 
                                self.latest_light = float(parts[3].split(':')[1].replace('lx', ''))
                    except: pass

            # Pull the absolute latest frame from the background thread
            if self.latest_camera_frame is None:
                time.sleep(0.01)
                continue
                
            # Copy it so we do not interfere with the thread's memory
            frame = self.latest_camera_frame.copy()
            frame = cv2.flip(frame, 1)
            frame_count += 1
            
            # Run inference on intervals to maintain performance
            if frame_count % SKIP_FRAMES == 0:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self._run_inference(rgb_frame)
                self.calculate_logic()

            # Render overlay text
            cv2.putText(frame, f"STRESS: {self.current_score}%", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255) if self.current_score>50 else (0,255,0), 3)
            
            # Compress to JPEG for the web feed
            ret, buffer = cv2.imencode('.jpg', frame)
            if ret:
                self.latest_frame_jpeg = buffer.tobytes()

            time.sleep(0.01)

    def _run_inference(self, frame):
        H, W, _ = frame.shape
        person_present = False
        
        f_res = self.mp_face.process(frame)
        if f_res.detections:
            d = f_res.detections[0]
            bb = d.location_data.relative_bounding_box
            x, y = int(bb.xmin * W), int(bb.ymin * H)
            w, h = int(bb.width * W), int(bb.height * H)
            
            if w > (W * 0.08) and x >= 0 and y >= 0 and (x+w) <= W and (y+h) <= H:
                person_present = True
                if self.emo_model:
                    f_img = frame[y:y+h, x:x+w]
                    self.latest_emo, _ = self.emo_model.predict_emotions(f_img, logits=False)
            else:
                self.latest_emo = "Away"
        else:
            self.latest_emo = "Away"

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
            self.pos_hist.clear()

    def _read_camera_frames(self):
        # Continuously pull frames to keep the buffer empty
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                self.latest_camera_frame = frame
            else:
                time.sleep(0.01)

    def get_state(self):
        return {
            "score": self.current_score,
            "prediction": self.care_prediction,
            "pred_color": self.pred_color,
            "temp": f"{self.latest_temp}°C",
            "hum": f"{self.latest_hum}%",
            "noise": f"{self.latest_noise:.1f} dB",
            "light": f"{self.latest_light:.0f} lx",
            "posture": self.latest_pos,
            "emotion": self.latest_emo,
            "recs": self.current_recs if self.current_recs else [("Condition Nominal", "Continue current workflow.")],
            "hardware_status": self.hardware_status,
            "logs": "\n".join(self.event_logs)
        }