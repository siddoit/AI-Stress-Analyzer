import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import class_weight
from tensorflow.keras.layers import Dense, Dropout, Input, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# --- 1. DATA LOADING & CLEANING ---
print("--- 1. Loading & Cleaning Data ---")
csv_file = 'posture_dataset.csv'

if not os.path.exists(csv_file):
    print("Error: CSV not found. Record data first.")
    exit()

df = pd.read_csv(csv_file)

# Force numeric & Clean garbage
df = df.apply(pd.to_numeric, errors='coerce')
df = df.dropna()
valid_classes = [0, 1, 2]
df = df[df['class'].isin(valid_classes)]
df['class'] = df['class'].astype(int)

# Check data balance
print("\nDataset Balance:")
print(df['class'].value_counts())

X = df.drop('class', axis=1).values
y = df['class'].values

# Split (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# --- 2. SYNTHETIC COORDINATE AUGMENTATION ---
print("\n--- 2. Applying Synthetic Coordinate Augmentation ---")

# Add 1% random noise to create fake variations
noise_factor = 0.01
noise = np.random.normal(0, noise_factor, X_train.shape)
X_train_augmented = X_train + noise
y_train_augmented = y_train

# Combine
X_train_final = np.concatenate((X_train, X_train_augmented))
y_train_final = np.concatenate((y_train, y_train_augmented))

print(f"Original samples: {len(X_train)}")
print(f"Augmented samples: {len(X_train_final)}")

# --- 3. CLASS WEIGHTS ---
print("\n--- 3. Calculating Class Weights ---")
weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train_final),
    y=y_train_final
)
class_weights = dict(enumerate(weights))
print(f"Weights Applied: {class_weights}") 

# --- 4. MODEL ARCHITECTURE ---
model = Sequential([
    Input(shape=(68,)),
    
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.4),

    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),

    Dense(3, activation='softmax')
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

# --- 5. CALLBACKS (FIXED) ---
callbacks = [
    EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True, verbose=1),
    
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, verbose=1),
    
    # FIXED: Typo removed 'val_val_accuracy' -> 'val_accuracy'
    # Added mode='max' so Keras knows higher is better
    ModelCheckpoint('my_posture_model.h5', monitor='val_accuracy', mode='max', save_best_only=True, verbose=1)
]

# --- 6. TRAIN ---
print("\n--- 6. Starting Training ---")
history = model.fit(
    X_train_final, y_train_final,
    validation_data=(X_test, y_test),
    epochs=100,
    batch_size=32,
    class_weight=class_weights,
    callbacks=callbacks,
    verbose=1
)

# --- 7. GRAPHS & DATA REPORTING ---
print("\n--- Final Performance Report ---")

# A. Evaluate
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Final Test Accuracy: {acc*100:.2f}%")

# B. Predictions
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)
class_names = ['Good', 'Slouch', 'Lean']

# C. Detailed Stats (F1, Precision, Recall)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=class_names))

# D. Plotting
plt.figure(figsize=(14, 5))

# Plot 1: Accuracy & Loss
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Score')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss (Lower is Better)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# E. Confusion Matrix Heatmap
plt.figure(figsize=(6, 5))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted AI Label')
plt.ylabel('Actual Label')
plt.title('Where did the AI make mistakes?')
plt.show()