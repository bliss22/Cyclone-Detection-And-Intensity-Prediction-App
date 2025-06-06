# utils/prediction_pipeline.py

import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.losses import MeanSquaredError
from ultralytics import YOLO
import os
from datetime import datetime
from pathlib import PosixPath
import streamlit as st
import torch

output_dir = "static"
os.makedirs(output_dir, exist_ok=True)


def test_path():
    import pathlib
    pathlib.PosixPath=pathlib.WindowsPath if os.name=='nt' else pathlib.PosixPath

test_path()
# Load YOLOv5 model properly from local path
@st.cache_resource
def load_model():
    return torch.hub.load('yolov5', 'custom', path='models\\best.pt', source='local')  # make sure 'yolov5' folder is in the same directory

model4= load_model() #object detection

# Load models
model1 = tf.keras.models.load_model('models/cyclone_detector.h5')  # binary
model2 = tf.keras.models.load_model(
    'models/cyclone_windspeed_cat.h5',
    custom_objects={'mse': MeanSquaredError()}
)
model3 = tf.keras.models.load_model(
    'models/cyclone_intensity.h5',
    custom_objects={'mse': MeanSquaredError()}
)


# Binary Classification Preprocessing
def preprocess_for_binary(image_path):
    img = Image.open(image_path).convert('L')
    img = img.resize((128, 128))
    img = np.array(img) / 255.0
    img = img[..., np.newaxis]
    return np.expand_dims(img, axis=0)

# Windspeed Prediction Preprocessing
def preprocess_for_windspeed(image_path):
    img = Image.open(image_path).convert('L')
    img = np.array(img)
    side = 50
    s = (img.shape[0] - side) // 2
    img = img[s:s+side, s:s+side]
    img = img / 255.0
    img = img[..., np.newaxis]
    return np.expand_dims(img, axis=0)



def preprocess_for_intensity(image_path):
    img = Image.open(image_path).convert('L').resize((75, 75))
    img = np.array(img) / 255.0
    img = np.stack([img, img], axis=-1).reshape(1, 75, 75, 2) # duplicate grayscale to (75, 75, 2)
    return img


# Main Pipeline
def run_pipeline(image_path):
    # Step 1: Binary classification
    binary_input = preprocess_for_binary(image_path)
    presence = model1.predict(binary_input)[0][0]

    if presence < 0.5:
        return {"cyclone_present": False}

    # Step 2: Object Detection with YOLOv5 (torch.hub)
    results = model4(image_path)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(output_dir, f"results_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)

    # Render the result (draw bounding boxes)
    annotated_img = results.render()[0]  # numpy array (BGR)

    # Save the annotated image
    annotated_image_path = os.path.join(save_dir, "annotated.jpg")
    from cv2 import imwrite, cvtColor, COLOR_BGR2RGB
    imwrite(annotated_image_path, annotated_img)

    # Extract bounding boxes
    try:
        boxes = results.xyxy[0].cpu().numpy().tolist()
    except Exception as e:
        boxes = []
        print("Bounding box extraction error:", e)


    # Step 3: Windspeed prediction
    wind_input = preprocess_for_windspeed(image_path)
    windspeed = float(model2.predict(wind_input)[0][0])

    def get_intensity_category(windspeed_knots):
        if windspeed_knots < 17:
            return "No Cyclone"
        elif 17 <= windspeed_knots <= 27:
            return "Depression"
        elif 28 <= windspeed_knots <= 33:
            return "Deep Depression"
        elif 34 <= windspeed_knots <= 47:
            return "Cyclonic Storm"
        elif 48 <= windspeed_knots <= 63:
            return "Severe Cyclonic Storm"
        elif 64 <= windspeed_knots <= 89:
            return "Very Severe Cyclonic Storm"
        elif 90 <= windspeed_knots <= 119:
            return "Extremely Severe Cyclonic Storm"
        else:
            return "Super Cyclonic Storm"
    intensity = get_intensity_category(windspeed)



    return {
        "cyclone_present": True,
        "bounding_boxes": boxes,
        "annotated_image": annotated_image_path,
        "windspeed": round(windspeed, 2),
        "intensity": intensity
    }
