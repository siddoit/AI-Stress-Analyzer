import cv2
import torch
import numpy as np
import mediapipe as mp
import os
import matplotlib.pyplot as plt      # <--- NEW
import seaborn as sns                # <--- NEW
from sklearn.metrics import classification_report, confusion_matrix # <--- NEW

# --- 🛠️ MONKEY PATCH (Fix PyTorch 2.6) ---
_original_load = torch.load
def fixed_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_load(*args, **kwargs)
torch.load = fixed_load
# -----------------------------------------

from hsemotion.facial_emotions import HSEmotionRecognizer

# 1. SETUP LOCAL BRAIN
print("🚀 Loading Local Brain...")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_file = 'enet_b0_8_best_vgaf.pt' 

if not os.path.exists(model_file):
    print(f"❌ ERROR: Could not find '{model_file}' in this folder.")
    print("👉 Please copy it here from C:\\Users\\sidha\\.hsemotion\\")
    exit()

try:
    model = HSEmotionRecognizer(model_name=model_file, device=device)
    print(f"✅ Local Brain Active on: {device.upper()}")
except Exception as e:
    print(f"❌ Brain Error: {e}")
    try:
        model = HSEmotionRecognizer(model_name='enet_b0_8_best_vgaf', device=device)
        print("⚠️ Loaded from global cache instead.")
    except:
        exit()

# 2. SETUP EYES (MediaPipe)
mp_face_detection = mp.solutions.face_detection
detector = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# ==========================================
# 📊 NEW: METRICS & RECORDING SETUP
# ==========================================
print("\n--- INSTRUCTIONS FOR METRICS ---")
print("1. Press 'n' and hold a NEUTRAL face to test Neutral accuracy.")
print("2. Press 'h' and hold a HAPPY face to test Happy accuracy.")
print("3. Press 's' and hold a SAD face to test Sad accuracy.")
print("4. Press 'a' and hold an ANGRY face to test Angry accuracy.")
print("5. Press 'r' (Report) to STOP recording and see Graphs/Scores.")
print("--------------------------------")

true_labels = []    # Ground Truth (What you promised to act)
pred_labels = []    # Prediction (What AI saw)
is_recording = False
target_emotion = "" # The emotion you are currently acting out

# List of classes your model outputs (Standard for HSEmotion)
valid_classes = ['Anger', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

while True:
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.flip(frame, 1)
    h_img, w_img, _ = frame.shape

    # 3. DETECT
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = detector.process(rgb_frame)

    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            
            x = int(bboxC.xmin * w_img)
            y = int(bboxC.ymin * h_img)
            w = int(bboxC.width * w_img)
            h = int(bboxC.height * h_img)

            x, y = max(0, x), max(0, y)
            if x + w > w_img: w = w_img - x
            if y + h > h_img: h = h_img - y

            if w < 20 or h < 20: continue

            # 4. PREDICT
            try:
                face_img = frame[y:y+h, x:x+w]
                face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                
                # 'emotion' is the string (e.g., 'Happy')
                emotion, scores = model.predict_emotions(face_rgb, logits=False)

                # --- 📊 NEW: RECORDING LOGIC ---
                if is_recording:
                    true_labels.append(target_emotion)
                    pred_labels.append(emotion) # Ensure strings match case (Happy vs happy)
                    
                    # Record indicator
                    cv2.circle(frame, (50, 50), 20, (0, 0, 255), -1)
                    cv2.putText(frame, f"REC: {target_emotion.upper()}", (80, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                # -------------------------------

                # Color logic
                color = (0, 255, 0)
                if emotion in ['Anger', 'Disgust', 'Fear', 'Sad']: color = (0, 0, 255)
                if emotion in ['Happy', 'Surprise']: color = (0, 255, 255)

                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.rectangle(frame, (x, y-35), (x+150, y), color, -1)
                cv2.putText(frame, emotion.upper(), (x+5, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2)
                
            except Exception:
                pass

    cv2.imshow('Sid Local AI', frame)
    
    # --- ⌨️ CONTROLS ---
    key = cv2.waitKey(1)
    
    if key == 27: # ESC
        break
    
    # Toggle Recording Types
    elif key == ord('n'): # Neutral
        is_recording = True
        target_emotion = 'Neutral'
    elif key == ord('h'): # Happy
        is_recording = True
        target_emotion = 'Happy'
    elif key == ord('s'): # Sad
        is_recording = True
        target_emotion = 'Sad'
    elif key == ord('a'): # Anger
        is_recording = True
        target_emotion = 'Anger'
        
    # Generate Report
    elif key == ord('r'):
        if not true_labels:
            print("No data recorded! Hold a key (n, h, s, a) first.")
        else:
            is_recording = False
            print("\n" + "="*40)
            print("      GENERATING PERFORMANCE REPORT      ")
            print("="*40)
            
            # 1. TEXT REPORT
            try:
                print(classification_report(true_labels, pred_labels))
            except:
                print("Not enough varied data to generate full report yet.")

            # 2. CONFUSION MATRIX (GRAPH)
            # Filter classes to only show emotions actually recorded + predicted
            unique_labels = sorted(list(set(true_labels + pred_labels)))
            cm = confusion_matrix(true_labels, pred_labels, labels=unique_labels)
            
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=unique_labels, yticklabels=unique_labels)
            plt.title('Emotion AI Accuracy Test')
            plt.xlabel('AI Predicted')
            plt.ylabel('Real Target (What you acted)')
            plt.tight_layout()
            plt.show() # Pops up window
            
            # Reset
            true_labels = []
            pred_labels = []
            print("Report done. Data reset.")

cap.release()
cv2.destroyAllWindows()