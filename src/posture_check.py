import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

# Load YOUR custom model
model = tf.keras.models.load_model('my_posture_model.h5')
classes = ['Good Posture', 'Slouching', 'Leaning/Bad']
colors = [(0, 255, 0), (0, 0, 255), (0, 255, 255)] # Green, Red, Yellow

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    
    # Pre-processing
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(img_rgb)
    
    status_text = "No Pose Detected"
    color = (255, 0, 0)
    confidence = 0

    if results.pose_landmarks:
        # 1. Extract the same 68 coordinates we trained on
        lm_list = []
        for i in range(17): # Only upper body (first 17 points)
            lm = results.pose_landmarks.landmark[i]
            lm_list.extend([lm.x, lm.y, lm.z, lm.visibility])
        
        # 2. Reshape for TensorFlow (1 sample, 68 features)
        input_data = np.array(lm_list).reshape(1, -1)
        
        # 3. Predict using the .h5 model
        prediction = model.predict(input_data, verbose=0)
        class_id = np.argmax(prediction)
        confidence = np.max(prediction)
        
        status_text = classes[class_id]
        color = colors[class_id]

        # Draw Skeleton
        mp.solutions.drawing_utils.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # UI Overlay
    cv2.rectangle(frame, (0,0), (350, 60), (0,0,0), -1)
    cv2.putText(frame, f"{status_text} ({int(confidence*100)}%)", 
                (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    
    cv2.imshow('Custom AI Posture Check', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()