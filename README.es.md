<!--
---
title: Flowing Vision - Detección de Fugas con IA
emoji: 💧
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
license: mit
app_port: 7860
---
-->

# Flowing Vision

![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-000000?logo=flask&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?logo=opencv&logoColor=white)
![OpenVINO](https://img.shields.io/badge/OpenVINO-0071C5?logo=intel&logoColor=white)
![Deploy](https://img.shields.io/badge/Hugging%20Face-black?logo=huggingface&logoColor=yellow)
![Docker](https://img.shields.io/badge/Docker-2496ED?logo=docker&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-brown)

Una aplicación sencilla que detecta fugas de agua utilizando visión por computadora. Creé esta aplicación durante un hackathon en Monterrey porque las fugas de agua son un gran problema allí. Están lidiando con sequías y las fugas de agua pueden causar miles de pesos en daños si no se detectan a tiempo.

[![Demo en Vivo](https://img.shields.io/badge/Prueba%20la%20Demo%20en%20Vivo%20aquí-yellow?logo=huggingface&style=for-the-badge&logoColor=white)](https://huggingface.co/spaces/LeonardoCerv/flowing-vision)

## Qué hace

Esta aplicación te ayuda a detectar fugas de agua utilizando visión por computadora:

- **Transmisión en vivo** - Apunta tu cámara web a las tuberías y recibe alertas instantáneas.
- **Análisis de fotos** - Sube una foto y te dirá si hay una fuga.

## Cómo funciona

### Cámara en Vivo
Activa tu cámara web y dirígela hacia tuberías o áreas propensas a fugas. La aplicación procesa cada cuadro de video en tiempo real utilizando un modelo de visión por computadora entrenado. Si el modelo detecta una fuga, muestra una alerta inmediatamente en la interfaz web.

### Subida de Fotos
Puedes subir una foto (PNG, JPG, etc.). La aplicación analiza la imagen con el mismo modelo de detección, buscando indicadores de fugas. Devuelve un resultado con un puntaje de confianza, para que puedas ver qué tan probable es que haya una fuga.

### El Modelo de IA
Utilicé modelos de visión por computadora que entrené personalmente en un pequeño conjunto de datos (100 imágenes) para reconocer fugas de agua. Optimicé el modelo utilizando OpenVINO, que es excelente para CPUs Intel.

## Por qué lo construí

Creé esta aplicación durante HackMTY porque Monterrey tiene serios problemas de agua, pero realmente esto podría ayudar a cualquiera. La idea es simple: detectar fugas temprano antes de que se conviertan en desastres.

Todo puede ejecutarse en tu propia computadora, por lo que tus fotos y videos permanecen privados. Si solo quieres probarlo, siempre puedes usar la demo web desplegada.

## Cómo empezar

### Qué necesitas
- Python 3.8 o más reciente
- Una cámara web (si deseas detección en vivo)

### Cómo instalar

```bash
# Descarga el código
git clone https://github.com/yourusername/flowing-vision.git
cd flowing-vision

# Configura un entorno virtual (recomendado)
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instala todo lo necesario
pip install -r requirements.txt

# Inicia la aplicación
python app.py
```