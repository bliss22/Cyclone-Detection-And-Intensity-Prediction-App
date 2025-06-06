# 🌪️ Cyclone Detection and Intensity Prediction App

A powerful web application that detects cyclones in satellite imagery, predicts their intensity, and estimates associated windspeed — built using deep learning and Streamlit.

---

## 🚀 Features

- ✅ **Cyclone Detection** from satellite images using YOLOv5 object detection.
- 📊 **Cyclone Intensity Prediction** using CNN-based regression models.
- 🌬️ **Windspeed Estimation** based on intensity class mapping.
- 🎯 **Visual Interface** with annotated image display and intuitive results.

---

## 🧠 Models Used

| Task                        | Model Used       | Description                               |
|----------------------------|------------------|-------------------------------------------|
| Cyclone Detection          | YOLOv5           | Bounding box detection on satellite images |
| Binary Classification      | CNN              | Cyclone vs Non-cyclone classification     |
| Intensity Prediction       | CNN Regression   | Maps image features to intensity class    |
| Windspeed Estimation       | Rule-based       | Based on IMD scale & model output         |

---

## 🗃️ Dataset

- Collected from [INSAT3D] and other open satellite archives.
- Includes preprocessed grayscale RGB and IR satellite images.
- Annotated with bounding boxes and intensity classes.


---

## 🛠️ Installation

```bash
git clone https://github.com/bliss22/Cyclone-Detection-And-Intensity-Prediction-App.git
cd Cyclone-Detection-And-Intensity-Prediction-App
pip install -r requirements.txt
streamlit run app.py
