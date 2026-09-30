"""
Speech Emotion Recognition (SER)
================================
Objective : Recognize human emotions (happy, angry, sad, neutral, etc.) from speech audio.
Approach  : MFCC-based feature extraction + Deep Learning (CNN + LSTM hybrid).
Datasets  : RAVDESS (default), TESS, EMO-DB (Berlin) — all supported via the LOADER
            functions below. Just point DATASET_PATH at the folder you downloaded.
"""

import os
import glob
import json
import pickle
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv1D, MaxPooling1D, BatchNormalization, Dropout,
    LSTM, Dense, Flatten, Input
)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

DATASET_NAME = "RAVDESS"
DATASET_PATH = "./data/RAVDESS"
SAMPLE_RATE = 22050
DURATION = 3
N_MFCC = 40
RANDOM_STATE = 42
BATCH_SIZE = 32
EPOCHS = 60
MODEL_OUT = "ser_cnn_lstm_model.h5"
ENCODER_OUT = "label_encoder.pkl"
SCALER_OUT = "feature_scaler.pkl"

RAVDESS_EMOTION_MAP = {
    "01": "neutral", "02": "calm", "03": "happy", "04": "sad",
    "05": "angry", "06": "fearful", "07": "disgust", "08": "surprised"
}

EMODB_EMOTION_MAP = {
    "W": "angry", "L": "boredom", "E": "disgust", "A": "fearful",
    "F": "happy", "T": "sad", "N": "neutral"
}


def load_ravdess(path):
    files, labels = [], []
    for f in glob.glob(os.path.join(path, "**", "*.wav"), recursive=True):
        parts = os.path.basename(f).split("-")
        if len(parts) >= 3 and parts[2] in RAVDESS_EMOTION_MAP:
            files.append(f)
            labels.append(RAVDESS_EMOTION_MAP[parts[2]])
    return files, labels


def load_tess(path):
    files, labels = [], []
    for f in glob.glob(os.path.join(path, "**", "*.wav"), recursive=True):
        stem = os.path.splitext(os.path.basename(f))[0].lower()
        emotion = stem.split("_")[-1]
        emotion = "surprised" if emotion == "ps" else emotion
        files.append(f)
        labels.append(emotion)
    return files, labels


def load_emodb(path):
    files, labels = [], []
    for f in glob.glob(os.path.join(path, "**", "*.wav"), recursive=True):
        code = os.path.basename(f)[5]
        if code in EMODB_EMOTION_MAP:
            files.append(f)
            labels.append(EMODB_EMOTION_MAP[code])
    return files, labels


def load_dataset(name, path):
    if name.upper() == "RAVDESS":
        return load_ravdess(path)
    elif name.upper() == "TESS":
        return load_tess(path)
    elif name.upper() == "EMODB":
        return load_emodb(path)
    else:
        raise ValueError(f"Unknown dataset: {name}")


def extract_features(file_path, sr=SAMPLE_RATE, duration=DURATION, n_mfcc=N_MFCC):
    try:
        y, sr = librosa.load(file_path, sr=sr, duration=duration)
        target_len = sr * duration
        if len(y) < target_len:
            y = np.pad(y, (0, target_len - len(y)))
        else:
            y = y[:target_len]

        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        mfcc_delta = librosa.feature.delta(mfcc)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        mel = librosa.feature.melspectrogram(y=y, sr=sr)
        mel_db = librosa.power_to_db(mel)
        zcr = librosa.feature.zero_crossing_rate(y)
        rms = librosa.feature.rms(y=y)

        feature_vector = np.concatenate([
            np.mean(mfcc, axis=1), np.std(mfcc, axis=1),
            np.mean(mfcc_delta, axis=1),
            np.mean(chroma, axis=1),
            np.mean(mel_db, axis=1),
            np.mean(zcr, axis=1),
            np.mean(rms, axis=1),
        ])
        return feature_vector
    except Exception as e:
        print(f"  [skip] {file_path}: {e}")
        return None


def build_feature_dataframe(files, labels):
    rows, valid_labels = [], []
    for i, (f, lbl) in enumerate(zip(files, labels)):
        feats = extract_features(f)
        if feats is not None:
            rows.append(feats)
            valid_labels.append(lbl)
        if (i + 1) % 50 == 0:
            print(f"  processed {i+1}/{len(files)} files")
    X = np.array(rows)
    y = np.array(valid_labels)
    return X, y


def build_model(input_shape, num_classes):
    model = Sequential([
        Input(shape=input_shape),

        Conv1D(256, kernel_size=5, padding="same", activation="relu"),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),

        Conv1D(128, kernel_size=5, padding="same", activation="relu"),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),

        LSTM(128, return_sequences=True),
        Dropout(0.3),
        LSTM(64),
        Dropout(0.3),

        Dense(64, activation="relu"),
        Dropout(0.3),
        Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    print(f"Loading dataset '{DATASET_NAME}' from {DATASET_PATH} ...")
    files, labels = load_dataset(DATASET_NAME, DATASET_PATH)
    print(f"Found {len(files)} audio files across {len(set(labels))} emotion classes: {sorted(set(labels))}")

    if len(files) == 0:
        print("No files found. Check DATASET_PATH and folder structure. Exiting.")
        return

    print("Extracting features (MFCC, chroma, mel, ZCR, RMS)...")
    X, y = build_feature_dataframe(files, labels)
    print(f"Feature matrix shape: {X.shape}")

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    y_categorical = to_categorical(y_encoded)
    num_classes = y_categorical.shape[1]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_reshaped = X_scaled[..., np.newaxis]

    X_train, X_test, y_train, y_test = train_test_split(
        X_reshaped, y_categorical, test_size=0.2,
        random_state=RANDOM_STATE, stratify=y_categorical
    )

    print(f"Train samples: {X_train.shape[0]} | Test samples: {X_test.shape[0]}")

    model = build_model(input_shape=(X_train.shape[1], 1), num_classes=num_classes)
    model.summary()

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )

    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_test, axis=1)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="weighted")
    print(f"\nTest Accuracy: {acc*100:.2f}%")
    print(f"Weighted F1-score: {f1:.4f}\n")
    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=le.classes_))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix - Speech Emotion Recognition")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    print("Saved confusion_matrix.png")

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"], label="train")
    plt.plot(history.history["val_accuracy"], label="val")
    plt.title("Accuracy"); plt.xlabel("Epoch"); plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], label="train")
    plt.plot(history.history["val_loss"], label="val")
    plt.title("Loss"); plt.xlabel("Epoch"); plt.legend()
    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=150)
    print("Saved training_curves.png")

    model.save(MODEL_OUT)
    with open(ENCODER_OUT, "wb") as f:
        pickle.dump(le, f)
    with open(SCALER_OUT, "wb") as f:
        pickle.dump(scaler, f)
    print(f"\nSaved model to {MODEL_OUT}")
    print(f"Saved label encoder to {ENCODER_OUT}")
    print(f"Saved feature scaler to {SCALER_OUT}")


def predict_emotion(file_path, model_path=MODEL_OUT,
                     encoder_path=ENCODER_OUT, scaler_path=SCALER_OUT):
    model = tf.keras.models.load_model(model_path)
    with open(encoder_path, "rb") as f:
        le = pickle.load(f)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    feats = extract_features(file_path)
    if feats is None:
        return None
    feats_scaled = scaler.transform(feats.reshape(1, -1))
    feats_reshaped = feats_scaled[..., np.newaxis]

    probs = model.predict(feats_reshaped)[0]
    pred_idx = np.argmax(probs)
    pred_label = le.inverse_transform([pred_idx])[0]

    result = {le.classes_[i]: float(probs[i]) for i in range(len(le.classes_))}
    print(f"Predicted emotion: {pred_label}")
    print("Class probabilities:", json.dumps(result, indent=2))
    return pred_label, result


if __name__ == "__main__":
    main()
    # Example single-file inference after training:
    # predict_emotion("path/to/new_sample.wav")
