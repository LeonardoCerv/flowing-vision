<!--
---
title: Flowing Vision - AI Leak Detection
emoji: 💧
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
license: mit
app_port: 7860
---
-->

> 📖 🇪🇸 También disponible en español: [README.es.md](README.es.md)

# Flowing Vision

![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-000000?logo=flask&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?logo=opencv&logoColor=white)
![OpenVINO](https://img.shields.io/badge/OpenVINO-0071C5?logo=intel&logoColor=white)
![Deploy](https://img.shields.io/badge/Hugging%20Face-black?logo=huggingface&logoColor=yellow)
![Docker](https://img.shields.io/badge/Docker-2496ED?logo=docker&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-brown)

A simple app that spots water leaks using computer vision. I built this during a hackathon in Monterrey because water leaks are a huge problem there, theyre dealing with drought and water leaks can cost thousands in damage if you don't catch them early.

[![Live Demo](https://img.shields.io/badge/Check%20out%20the%20Live%20Demo%20here-yellow?logo=huggingface&style=for-the-badge&logoColor=white)](https://huggingface.co/spaces/LeonardoCerv/flowing-vision)

## What it does

This app helps you catch water leaks using computer vision:

- **Live camera feed** - Point your webcam at pipes and get instant alerts
- **Photo analysis** - Upload a picture and it'll tell you if there's a leak

## How it works

### Live Camera
Activate your webcam and direct it at pipes or areas prone to leaks. The app processes each video frame in real time using a trained computer vision model. If the model detects a leak, it immediately displays an alert in the web interface.

### Photo Upload
You can upload a photo (PNG, JPG, etc.). The app runs the image through the same detection model, analyzing for leak indicators. It returns a result with a confidence score, so you can see how likely it is that a leak is present.

### The AI Model
I used computer vision models that I personally trained on a small dataset (100 images) to recognize water leaks. I optimized the model using open vino which is great for intel CPUs.

## Why I built this

I built this during HackMTY because Monterrey has serious water issues, but really this could help anyone. The idea is simple: catch leaks early before they turn into disasters.

Everything can run on your own computer, so your photos and video stay private. If you just wanna test it out you can always try out the deployed web demo.

## Getting started

### What you need
- Python 3.8 or newer
- A webcam (if you want live detection)

### How to install

```bash
# Download the code
git clone https://github.com/yourusername/flowing-vision.git
cd flowing-vision

# Set up a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install everything you need
pip install -r requirements.txt

# Start the app
python app.py
```