import cv2
import mediapipe as mp
import csv
import numpy as np
import os  # <--- NEW: To check if file exists

# Setup
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
cap = cv2.VideoCapture(0)

csv_file = 'posture_dataset.csv'

# CSV Header (34 inputs: 17 landmarks * x,y coordinates)
landmarks = ['class']
for val in range(0, 17): 
    landmarks += [f'x{val}', f'y{val}', f'z{val}', f'v{val}']

# --- NEW LOGIC: Only create headers if file is new ---
if not os.path.isfile(csv_file):
    print(f"Creating new file: {csv_file}")
    with open(csv_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(landmarks)
else:
    print(f"File found. Appending to: {csv_file}")

print("--- DATA COLLECTOR (APPEND MODE) ---")
print("Hit 'g' to record GOOD Posture frame")
print("Hit 'b' to record SLOUCHING frame")
print("Hit 'l' to record LEANING frame")
print("Hit 'q' to quit")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    
    # Draw skeleton just for visuals
    if results.pose_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    cv2.imshow('Data Collector', frame)
    k = cv2.waitKey(1)
    
    if k == ord('q'): break
    
    # Only save if human detected and key pressed
    if results.pose_landmarks and k in [ord('g'), ord('b'), ord('l')]:
        # Extract upper body pose coordinates
        row = []
        
        # Label
        if k == ord('g'): row.append(0) # 0 = Good
        if k == ord('b'): row.append(1) # 1 = Slouch
        if k == ord('l'): row.append(2) # 2 = Leaning/Other
        
        # Extract landmarks (first 17 are upper body)
        lms = results.pose_landmarks.landmark
        for i in range(17):
            row.append(lms[i].x)
            row.append(lms[i].y)
            row.append(lms[i].z)
            row.append(lms[i].visibility)
            
        # --- WRITE MODE: 'a' for APPEND ---
        with open(csv_file, mode='a', newline='') as f:
            writer_object = csv.writer(f)
            writer_object.writerow(row)
        
        print(f"Recorded class: {row[0]} (Rows in CSV: {sum(1 for _ in open(csv_file)) - 1})")

cap.release()
cv2.destroyAllWindows()