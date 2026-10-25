# ============================================================
# CardioPredict: Multi-Modal Cardiovascular Disease Diagnosis
# Tabular Clinical Data + ECG Signal (1D-CNN) + ECG Image (2D-CNN)
# ============================================================

import os
import warnings
from PIL import Image
import cv2
import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

warnings.filterwarnings("ignore")

# ------------------------------------------------------------
# Dataset Paths Configuration (Local & Colab Compatible)
# ------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in locals() else os.getcwd()
DATASET_DIR = os.path.join(SCRIPT_DIR, "dataset")

if not os.path.exists(DATASET_DIR) and os.path.exists("/content/drive"):
    DATASET_DIR = "/content/drive/MyDrive/CardioPredict"

CLEV_PATH = os.path.join(DATASET_DIR, "heart.csv")
FRAM_PATH = os.path.join(DATASET_DIR, "framingham.csv")
ECG_TRAIN = os.path.join(DATASET_DIR, "ECG", "mitbih_train.csv")
ECG_TEST = os.path.join(DATASET_DIR, "ECG", "mitbih_test.csv")
IMG_DIR = os.path.join(DATASET_DIR, "ECG_Images")

# ------------------------------------------------------------
# 1. Cleveland Dataset Pipeline & Model Training
# ------------------------------------------------------------
print("Loading Cleveland dataset...")
if os.path.exists(CLEV_PATH):
    df_clev = pd.read_csv(CLEV_PATH)
else:
    try:
        from google.colab import files
        print("Upload Cleveland dataset...")
        uploaded = files.upload()
        df_clev = pd.read_csv(list(uploaded.keys())[0])
    except ImportError:
        raise FileNotFoundError(f"Cleveland dataset not found at {CLEV_PATH}")

if "target" in df_clev.columns:
    y_clev, X_clev = df_clev["target"], df_clev.drop(columns=["target"])
elif "num" in df_clev.columns:
    y_clev, X_clev = (df_clev["num"] > 0).astype(int), df_clev.drop(columns=["num"])
else:
    y_clev, X_clev = df_clev.iloc[:, -1], df_clev.iloc[:, :-1]

num_cols = [c for c in X_clev.columns if pd.api.types.is_numeric_dtype(X_clev[c])]
cat_cols = [c for c in X_clev.columns if c not in num_cols]

pipe_clev = Pipeline([
    ("pre", ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), num_cols),
        ("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore"))
        ]), cat_cols)
    ])),
    ("rf", RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced_subsample"))
])

Xtr_c, Xte_c, ytr_c, yte_c = train_test_split(
    X_clev, y_clev, test_size=0.2, random_state=42, stratify=y_clev
)
pipe_clev.fit(Xtr_c, ytr_c)
CLE_FEATURES = list(X_clev.columns)
print(f"Cleveland model trained successfully. Test accuracy: {pipe_clev.score(Xte_c, yte_c):.4f}")

# ------------------------------------------------------------
# 2. Framingham Dataset Pipeline & Model Training
# ------------------------------------------------------------
print("Loading Framingham dataset...")
if os.path.exists(FRAM_PATH):
    df_fram = pd.read_csv(FRAM_PATH)
else:
    try:
        from google.colab import files
        print("Upload Framingham dataset...")
        uploaded = files.upload()
        df_fram = pd.read_csv(list(uploaded.keys())[0])
    except ImportError:
        raise FileNotFoundError(f"Framingham dataset not found at {FRAM_PATH}")

if "TenYearCHD" in df_fram.columns:
    y_fram, X_fram = df_fram["TenYearCHD"], df_fram.drop(columns=["TenYearCHD"])
else:
    y_fram, X_fram = df_fram.iloc[:, -1], df_fram.iloc[:, :-1]

num_cols_f = [c for c in X_fram.columns if pd.api.types.is_numeric_dtype(X_fram[c])]
cat_cols_f = [c for c in X_fram.columns if c not in num_cols_f]

pipe_fram = Pipeline([
    ("pre", ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), num_cols_f),
        ("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore"))
        ]), cat_cols_f)
    ])),
    ("rf", RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced_subsample"))
])

Xtr_f, Xte_f, ytr_f, yte_f = train_test_split(
    X_fram, y_fram, test_size=0.2, random_state=42, stratify=y_fram
)
pipe_fram.fit(Xtr_f, ytr_f)
FRAM_FEATURES = list(X_fram.columns)
print(f"Framingham model trained successfully. Test accuracy: {pipe_fram.score(Xte_f, yte_f):.4f}")

# ------------------------------------------------------------
# 3. ECG Signal 1D-CNN (MIT-BIH Arrhythmia)
# ------------------------------------------------------------
print("Training 1D-CNN on MIT-BIH ECG signal dataset...")
tr = pd.read_csv(ECG_TRAIN, header=None, nrows=20000)
te = pd.read_csv(ECG_TEST, header=None, nrows=5000)

X_train_ecg, y_train_ecg = tr.iloc[:, :-1].values.astype("float32"), tr.iloc[:, -1].values.astype("int64")
X_test_ecg, y_test_ecg = te.iloc[:, :-1].values.astype("float32"), te.iloc[:, -1].values.astype("int64")

gmax = max(X_train_ecg.max(), X_test_ecg.max())
X_train_ecg, X_test_ecg = X_train_ecg / gmax, X_test_ecg / gmax
X_train_ecg = X_train_ecg.reshape((X_train_ecg.shape[0], X_train_ecg.shape[1], 1))
X_test_ecg = X_test_ecg.reshape((X_test_ecg.shape[0], X_test_ecg.shape[1], 1))

model_ecg = models.Sequential([
    layers.Conv1D(32, 5, activation="relu", input_shape=(X_train_ecg.shape[1], 1)),
    layers.MaxPooling1D(2),
    layers.Conv1D(64, 5, activation="relu"),
    layers.MaxPooling1D(2),
    layers.Flatten(),
    layers.Dense(64, activation="relu"),
    layers.Dense(5, activation="softmax")
])

model_ecg.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
model_ecg.fit(
    X_train_ecg, y_train_ecg,
    epochs=2,
    batch_size=128,
    validation_data=(X_test_ecg, y_test_ecg),
    verbose=1
)

ECG_CLASSES = ["Normal", "Supraventricular", "Ventricular", "Fusion", "Unknown"]

# ------------------------------------------------------------
# 4. ECG Image 2D-CNN (4-Class Image Classification)
# ------------------------------------------------------------
print("Training 2D-CNN on ECG images...")
img_size = 128
batch_size = 32

datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2
)

train_gen = datagen.flow_from_directory(
    IMG_DIR,
    target_size=(img_size, img_size),
    color_mode="grayscale",
    batch_size=batch_size,
    class_mode="sparse",
    subset="training"
)

val_gen = datagen.flow_from_directory(
    IMG_DIR,
    target_size=(img_size, img_size),
    color_mode="grayscale",
    batch_size=batch_size,
    class_mode="sparse",
    subset="validation"
)

model_img = models.Sequential([
    layers.Conv2D(16, (3, 3), activation="relu", input_shape=(img_size, img_size, 1)),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D(2, 2),
    layers.Flatten(),
    layers.Dense(64, activation="relu"),
    layers.Dense(4, activation="softmax")
])

model_img.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
model_img.fit(train_gen, validation_data=val_gen, epochs=10)

ECG_IMAGE_CLASSES = list(train_gen.class_indices.keys())
print("ECG Image Classes:", ECG_IMAGE_CLASSES)

DISEASE_MAP_IMG = {
    "Normal": ("Normal ECG (No abnormality)", "Low Risk", "Maintain healthy lifestyle."),
    "MI": ("Myocardial Infarction", "High Risk", "Cardiologist consultation recommended."),
    "Abnormal": ("General ECG Abnormality", "Medium Risk", "Consult a cardiologist for further evaluation."),
    "HistoryMI": ("Prior Myocardial Infarction History", "Medium Risk", "Regular clinical monitoring advised.")
}

# ------------------------------------------------------------
# 5. Prediction & Risk Fusion Functions
# ------------------------------------------------------------
def predict_tabular(patient_dict, dataset="cleveland"):
    if dataset == "cleveland":
        feats, model = CLE_FEATURES, pipe_clev
    else:
        feats, model = FRAM_FEATURES, pipe_fram
    row = {f: patient_dict.get(f, np.nan) for f in feats}
    prob = float(model.predict_proba(pd.DataFrame([row]))[0, 1])
    risk = "High Risk" if prob >= 0.7 else ("Medium Risk" if prob >= 0.4 else "Low Risk")
    return prob, risk

def predict_ecg_signal(arr):
    arr = np.array(arr, dtype="float32")
    arr = arr / np.max(arr) if np.max(arr) != 0 else arr
    arr = arr.reshape(1, -1, 1)
    probs = model_ecg.predict(arr, verbose=0)[0]
    k = int(np.argmax(probs))
    risk_map = {
        "Normal": "Low Risk",
        "Supraventricular": "Medium Risk",
        "Fusion": "High Risk",
        "Ventricular": "High Risk",
        "Unknown": "Medium Risk"
    }
    return ECG_CLASSES[k], risk_map[ECG_CLASSES[k]]

def predict_ecg_image(img_path):
    img = Image.open(img_path).convert("L").resize((img_size, img_size))
    arr = np.array(img).astype("float32") / 255.0
    arr = arr.reshape(1, img_size, img_size, 1)
    probs = model_img.predict(arr, verbose=0)[0]
    pred_class = ECG_IMAGE_CLASSES[np.argmax(probs)]
    disease, risk, rec = DISEASE_MAP_IMG[pred_class]
    return disease, risk, rec

def fuse_risk(tab_risk, ecg_risk):
    levels = {"Low Risk": 0, "Medium Risk": 1, "High Risk": 2}
    inv = {v: k for k, v in levels.items()}
    final = inv[max(levels[tab_risk], levels[ecg_risk])]
    rec = (
        "Maintain healthy lifestyle."
        if final == "Low Risk"
        else (
            "Schedule consultation soon."
            if final == "Medium Risk"
            else "Cardiologist consultation recommended."
        )
    )
    return final, rec

# ------------------------------------------------------------
# 6. Gradio Web Application
# ------------------------------------------------------------
def predict_ecg_image_web(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (img_size, img_size)) / 255.0
    arr = np.expand_dims(img, axis=(0, -1))
    pred = model_img.predict(arr, verbose=0)
    cls = ECG_IMAGE_CLASSES[int(np.argmax(pred))]
    disease, risk, rec = DISEASE_MAP_IMG[cls]
    return f"""ECG Image Analysis:
Diagnosis: {disease}
Risk Level: {risk}
Recommendation: {rec}"""

iface = gr.Interface(
    fn=predict_ecg_image_web,
    inputs=gr.Image(type="numpy", label="Upload ECG Image"),
    outputs="text",
    title="CardioPredict - ECG Image Diagnosis",
    description="Upload a 12-lead ECG image to classify heart condition (Normal, MI, Abnormal, HistoryMI) and determine risk level."
)

if __name__ == "__main__":
    iface.launch(share=True)
