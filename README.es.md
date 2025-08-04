# Flowing Vision

![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-000000?logo=flask&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?logo=opencv&logoColor=white)
![OpenVINO](https://img.shields.io/badge/OpenVINO-0071C5?logo=intel&logoColor=white)
![Field](https://img.shields.io/badge/Field-Computer%20Vision-white)
![License](https://img.shields.io/badge/License-MIT-brown)

Un sistema de detección de fugas impulsado por IA que utiliza visión por computadora para identificar fugas de agua en tiempo real a través de cámaras o imágenes subidas. Desarrollado originalmente durante HackMTY para abordar la crisis de agua en Monterrey, este proyecto ayuda a prevenir daños y reparaciones costosas mediante tecnología avanzada de detección de fugas.

[![Demo en Vivo](https://img.shields.io/badge/Visita%20la%20version%20en%20Vivo%20aqui-blue?style=for-the-badge&logoColor=white)](https://flowingvision.leonardocerv.hackclub.app)

## ¿Qué hace?

Detecta fugas de agua antes de que se conviertan en problemas costosos:

- **Detección en tiempo real** usando tu cámara o webcam
- **Análisis de imágenes subidas** para detección instantánea de fugas
- **Precisión impulsada por IA** usando modelos optimizados de visión por computadora
- **Procesamiento local** - tus datos nunca salen de tu dispositivo
- **Interfaz web** - no requiere instalación

## Características

### Detección en Vivo
Procesamiento de video en tiempo real usando tu cámara o webcam, con monitoreo continuo y alertas instantáneas. El sistema soporta seguimiento de sesiones, estadísticas y gestión de colas para múltiples usuarios. La comunicación entre el servidor y los clientes se maneja mediante WebSockets para una verdadera capacidad de respuesta en tiempo real.

### Subida de Imágenes
Analiza imágenes estáticas al instante para la detección de fugas. La plataforma soporta múltiples formatos de imagen (PNG, JPG, JPEG, GIF, BMP, TIFF) y proporciona resultados detallados con puntuaciones de confianza. Las imágenes subidas se eliminan automáticamente después de 30 segundos para garantizar la privacidad y la gestión eficiente de recursos.

### Modelo de IA
El motor de detección de fugas está impulsado por modelos avanzados de visión por computadora, optimizados con Intel OpenVINO para inferencia rápida. Soporta múltiples formatos de modelo (ONNX, PyTorch, OpenVINO) y ofrece alta precisión con puntuaciones de confianza. El sistema está diseñado para un rendimiento eficiente y en tiempo real en hardware de consumo.

## Sobre el Proyecto

El futuro de Flowing Vision es ayudar a fábricas e individuos a detectar proactivamente fugas de agua antes de que se conviertan en problemas costosos, usando visión por computadora. Al aprovechar la IA y la visión por computadora, la plataforma puede procesar tanto transmisiones de video en vivo como imágenes subidas, haciéndola adecuada para una amplia gama de casos de uso, desde monitoreo doméstico hasta aplicaciones industriales. Todo el procesamiento se realiza localmente, garantizando la privacidad y seguridad de los datos del usuario. El proyecto fue desarrollado originalmente durante HackMTY para abordar la crisis de agua en Monterrey, pero su tecnología es aplicable en cualquier lugar donde la conservación del agua y la prevención de daños sean prioridades.

## Inicio Rápido

### Requisitos Previos
- Python 3.8+
- Webcam (para detección en vivo)
- Navegador web moderno

### Instalación

```bash
# Clona el repositorio
git clone https://github.com/yourusername/flowing-vision.git
cd flowing-vision

# Crea un entorno virtual
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instala las dependencias
pip install -r requirements.txt

# Ejecuta el script de configuración (opcional)
python app.py
```
