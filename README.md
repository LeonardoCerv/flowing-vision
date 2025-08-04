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

> 📖 🇪🇸 También disponible en español: [README.es.md](README.es.md)

# Flowing Vision

![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-000000?logo=flask&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?logo=opencv&logoColor=white)
![OpenVINO](https://img.shields.io/badge/OpenVINO-0071C5?logo=intel&logoColor=white)
![Field](https://img.shields.io/badge/Field-Computer%20Vision-white)
![License](https://img.shields.io/badge/License-MIT-brown)

An AI-powered leak detection system that uses computer vision to identify water leaks in real-time through camera feeds or uploaded images. Originally developed during HackMTY to address Monterrey's water crisis, this project helps prevent water damage and costly repairs through advanced leak detection technology.

[![Live Demo](https://img.shields.io/badge/Check%20out%20the%20Live%20Demo%20here-blue?style=for-the-badge&logoColor=white)](https://flowingvision.leonardocerv.hackclub.app)

## What it does

Detect water leaks before they become expensive problems:

- **Real-time detection** using your camera or webcam
- **Image upload analysis** for instant leak detection
- **AI-powered accuracy** using optimized computer vision models
- **Local processing** - your data never leaves your device
- **Web-based interface** - no installation required


## Features

### Live Detection
Real-time video processing using your camera or webcam, with continuous monitoring and instant alerts. The system supports session tracking, statistics, and queue management for multiple users. Communication between the server and clients is handled via WebSockets for true real-time responsiveness.

### Image Upload
Analyze static images instantly for leak detection. The platform supports multiple image formats (PNG, JPG, JPEG, GIF, BMP, TIFF) and provides detailed detection results with confidence scores. Uploaded images are automatically cleaned up after 30 seconds to ensure privacy and efficient resource management.

### AI Model
The leak detection engine is powered by advanced computer vision models, optimized using Intel OpenVINO for fast inference. It supports multiple model formats (ONNX, PyTorch, OpenVINO) and delivers high accuracy with confidence scoring. The system is designed for efficient, real-time performance on consumer hardware.

## About the Project

Flowing Vision's future is to help factories and individuals proactively detect water leaks before they escalate into costly problems, using computer vision. By leveraging AI and computer vision, the platform can process both live video feeds and uploaded images, making it suitable for a wide range of use cases, from home monitoring to industrial applications. All processing is done locally, ensuring user privacy and data security. The project was originally developed during HackMTY to address Monterrey's water crisis, but its technology is applicable anywhere water conservation and damage prevention are priorities.

## Quick Start

### Prerequisites
- Python 3.8+
- Webcam (for live detection)
- Modern web browser

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/flowing-vision.git
cd flowing-vision

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run setup script (optional)
chmod +x setup.sh
./setup.sh