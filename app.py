#!/usr/bin/env python3
"""
Flask web application for real-time leak detection using webcam
"""

from flask import Flask, render_template, jsonify, request, send_from_directory
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
from openvino import Core
import base64
import time
from datetime import datetime
import os
from dotenv import load_dotenv
from pymongo import MongoClient
import threading
from werkzeug.utils import secure_filename

load_dotenv()

# Configuration
CONFIDENCE_THRESHOLD = 0.1  # Adjustable confidence threshold for leak detection

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}

socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Global variables for model and database
compiled_model = None
client = None
collection = None

# Session tracking and queue management
active_sessions = {}
live_detection_queue = []
MAX_LIVE_USERS = 5

def setup_mongodb():
    """Setup MongoDB connection"""
    try:
        uri = os.getenv('DATABASE_URL')
        if not uri:
            print("Warning: DATABASE_URL environment variable not found")
            return None, None
        
        client = MongoClient(uri, serverSelectionTimeoutMS=5000)
        db = client['hidro']
        collection = db['leaks']
        print('Connected to MongoDB')
        return client, collection
        
    except Exception as e:
        print(f"Error connecting to MongoDB: {str(e)}")
        print("Continuing without database connection...")
        return None, None

def load_openvino_model():
    """Load OpenVINO model"""
    try:
        ie = Core()
        model_path = os.path.join('AI_model', 'OpenVinoModel.xml')
        weights_path = os.path.join('AI_model', 'OpenVinoModel.bin')
        
        model = ie.read_model(model=model_path, weights=weights_path)
        compiled_model = ie.compile_model(model=model, device_name="CPU")
        print("OpenVINO model loaded successfully")
        return compiled_model
    except Exception as e:
        print(f"Error loading OpenVINO model: {str(e)}")
        return None

def encode_frame(frame):
    """Convert frame to base64 encoded string"""
    try:
        is_success, buffer = cv2.imencode(".jpg", frame)
        if not is_success:
            return None
        
        encoded = base64.b64encode(buffer).decode('utf-8')
        return encoded
    
    except Exception as e:
        print(f"Error encoding frame: {str(e)}")
        return None

def process_frame(frame_data, session_id):
    """Process a single frame for leak detection"""
    global compiled_model, collection
    
    if compiled_model is None:
        return None
    
    try:
        # Decode base64 frame
        frame_bytes = base64.b64decode(frame_data.split(',')[1])
        frame_np = np.frombuffer(frame_bytes, dtype=np.uint8)
        frame = cv2.imdecode(frame_np, cv2.IMREAD_COLOR)
        
        if frame is None:
            return None
        
        h, w, _ = frame.shape
        
        # Prepare model input
        frame_resized = cv2.resize(frame, (640, 640))
        input_data = np.expand_dims(frame_resized.transpose(2, 0, 1), axis=0).astype(np.float32) / 255.0
        
        # Model inference
        results = compiled_model([input_data])[compiled_model.output(0)]
        
        detections = []
        
        # Analyze confidence distribution for debugging
        confidences = [result[4] for result in results[0]]
        max_confidence = max(confidences) if confidences else 0
        print(f"Max confidence in real-time frame: {max_confidence:.4f}")
        
        # Use configurable confidence threshold
        best_confidence = CONFIDENCE_THRESHOLD
        
        for result in results[0]:
            confidence = float(result[4])
            if confidence > best_confidence:
                print(f"Real-time detection found with confidence: {confidence:.4f}")
                x1, y1, x2, y2 = map(int, result[:4])
                
                # Scale coordinates back to original frame size
                x1 = int(x1 * w / 640)
                y1 = int(y1 * h / 640)
                x2 = int(x2 * w / 640)
                y2 = int(y2 * h / 640)
                
                detection = {
                    'confidence': confidence,
                    'bbox': [x1, y1, x2, y2],
                    'timestamp': time.time()
                }
                detections.append(detection)
        
        # Update session tracking
        if session_id in active_sessions:
            session = active_sessions[session_id]
            
            if detections:
                session['leak_frames'] += 1
                session['leak_detections'].extend(detections)
                
                # Check if this is the first time we hit 20 frames for this continuous leak
                if session['leak_frames'] == 20:
                    best_detection = max(session['leak_detections'], key=lambda d: d['confidence'])
                    
                    leak_record = {
                        'timestamp': datetime.fromtimestamp(best_detection['timestamp']),
                        'accuracy': best_detection['confidence'],
                        'frames_detected': session['leak_frames'],
                        'bbox': best_detection['bbox'],
                        'screenshot': encode_frame(frame),
                        'leak_id': f"leak_{int(best_detection['timestamp'])}",
                        'session_id': session_id
                    }
                    
                    # Save to database if connected
                    if collection is not None:
                        try:
                            collection.insert_one(leak_record)
                            print(f"Leak {leak_record['leak_id']} saved to database")
                        except Exception as e:
                            print(f"Error saving leak to database: {str(e)}")
                    
                    session['confirmed_leaks'].append(leak_record)
                    print(f"Confirmed leak added: {session['leak_frames']} frames detected")

                # Emit updated session stats
                emit('detection_result', {
                    'detections': detections,
                    'session_stats': {
                        'leak_frames': session['leak_frames'],
                        'confirmed_leaks': len(session['confirmed_leaks'])
                    }
                })
            else:
                # Reset when no leak detected - this allows the counter to go up again for next leak
                session['leak_frames'] = 0
                session['leak_detections'] = []
        
        return {
            'detections': detections,
            'session_stats': {
                'leak_frames': active_sessions[session_id]['leak_frames'] if session_id in active_sessions else 0,
                'confirmed_leaks': len(active_sessions[session_id]['confirmed_leaks']) if session_id in active_sessions else 0
            }
        }
        
    except Exception as e:
        print(f"Error processing frame: {str(e)}")
        return None

def allowed_file(filename):
    """Check if the uploaded file has an allowed extension"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_static_image(image_path):
    """Process a static image for leak detection"""
    global compiled_model
    
    if compiled_model is None:
        return None
    
    try:
        # Read the image
        frame = cv2.imread(image_path)
        if frame is None:
            return None
        
        h, w, _ = frame.shape
        
        # Prepare model input
        frame_resized = cv2.resize(frame, (640, 640))
        input_data = np.expand_dims(frame_resized.transpose(2, 0, 1), axis=0).astype(np.float32) / 255.0
        
        # Model inference
        results = compiled_model([input_data])[compiled_model.output(0)]
        
        detections = []
        
        # Analyze confidence distribution for debugging
        confidences = [result[4] for result in results[0]]
        max_confidence = max(confidences) if confidences else 0
        print(f"Max confidence in static image: {max_confidence:.4f}")
        
        # Use configurable confidence threshold
        best_confidence = CONFIDENCE_THRESHOLD
        
        for result in results[0]:
            confidence = float(result[4])
            if confidence > best_confidence:
                print(f"Static image detection found with confidence: {confidence:.4f}")
                x1, y1, x2, y2 = map(int, result[:4])
                
                # Scale coordinates back to original frame size
                x1 = int(x1 * w / 640)
                y1 = int(y1 * h / 640)
                x2 = int(x2 * w / 640)
                y2 = int(y2 * h / 640)
                
                detection = {
                    'confidence': confidence,
                    'bbox': [x1, y1, x2, y2],
                    'timestamp': time.time()
                }
                detections.append(detection)
        
        # Create result with annotated image
        annotated_frame = frame.copy()
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            confidence = detection['confidence']
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            
            # Draw label
            label = f'Leak: {confidence:.1%}'
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(annotated_frame, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), (0, 0, 255), -1)
            cv2.putText(annotated_frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Save annotated image
        annotated_filename = f"annotated_{os.path.basename(image_path)}"
        annotated_path = os.path.join(app.config['UPLOAD_FOLDER'], annotated_filename)
        cv2.imwrite(annotated_path, annotated_frame)
        
        return {
            'detections': detections,
            'original_image': os.path.basename(image_path),
            'annotated_image': annotated_filename,
            'image_dimensions': {'width': w, 'height': h},
            'leak_detected': len(detections) > 0
        }
        
    except Exception as e:
        print(f"Error processing static image: {str(e)}")
        return None

@app.route('/')
def index():
    """Main page with both live detection and upload functionality"""
    return render_template('index.html')

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': compiled_model is not None,
        'database_connected': collection is not None,
        'active_live_users': len([s for s in active_sessions.values() if s.get('live_detection_active', False)]),
        'max_live_users': MAX_LIVE_USERS
    })

@app.route('/upload', methods=['POST'])
def upload_image():
    """Handle image upload and processing"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file and allowed_file(file.filename):
        try:
            # Ensure upload directory exists
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            
            # Save uploaded file
            filename = secure_filename(file.filename)
            timestamp = int(time.time())
            filename = f"{timestamp}_{filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Process the image
            result = process_static_image(filepath)
            
            if result is None:
                # Clean up uploaded file on error
                if os.path.exists(filepath):
                    os.remove(filepath)
                return jsonify({'error': 'Failed to process image'})
            
            # Save to database if connected
            if collection is not None and result['leak_detected']:
                try:
                    leak_record = {
                        'timestamp': datetime.now(),
                        'accuracy': max([d['confidence'] for d in result['detections']]),
                        'detections_count': len(result['detections']),
                        'image_file': result['original_image'],
                        'annotated_file': result['annotated_image'],
                        'image_dimensions': result['image_dimensions'],
                        'leak_id': f"static_leak_{timestamp}",
                        'type': 'static_image'
                    }
                    collection.insert_one(leak_record)
                    print(f"Static image leak {leak_record['leak_id']} saved to database")
                except Exception as e:
                    print(f"Error saving static image leak to database: {str(e)}")
            
            # Schedule file cleanup after response is sent
            def cleanup_files():
                try:
                    # Delete original uploaded file
                    if os.path.exists(filepath):
                        os.remove(filepath)
                        print(f"Deleted original file: {filepath}")
                    
                    # Delete annotated file
                    annotated_path = os.path.join(app.config['UPLOAD_FOLDER'], result['annotated_image'])
                    if os.path.exists(annotated_path):
                        os.remove(annotated_path)
                        print(f"Deleted annotated file: {annotated_path}")
                except Exception as e:
                    print(f"Error cleaning up files: {str(e)}")
            
            # Schedule cleanup after 30 seconds to allow the client to download/view
            threading.Timer(30.0, cleanup_files).start()
            
            return jsonify({
                'success': True,
                'result': result
            })
            
        except Exception as e:
            print(f"Error handling file upload: {str(e)}")
            # Clean up uploaded file on error
            if 'filepath' in locals() and os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({'error': 'Failed to process uploaded file'})
    
    return jsonify({'error': 'File type not allowed'})

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    session_id = request.sid
    active_sessions[session_id] = {
        'start_time': time.time(),
        'leak_frames': 0,
        'leak_detections': [],
        'confirmed_leaks': [],
        'live_detection_active': False,
        'in_queue': False
    }
    emit('connected', {'session_id': session_id})
    print(f"Client connected: {session_id}")

@socketio.on('request_live_detection')
def handle_request_live_detection():
    """Handle request to start live detection"""
    session_id = request.sid
    
    if session_id not in active_sessions:
        emit('live_detection_response', {'allowed': False, 'reason': 'Session not found'})
        return
    
    # Count current active live detection users
    active_live_users = len([s for s in active_sessions.values() if s.get('live_detection_active', False)])
    
    if active_live_users >= MAX_LIVE_USERS:
        # Add to queue
        if session_id not in live_detection_queue:
            live_detection_queue.append(session_id)
            active_sessions[session_id]['in_queue'] = True
        
        queue_position = live_detection_queue.index(session_id) + 1
        emit('live_detection_response', {
            'allowed': False, 
            'reason': 'queue_full',
            'queue_position': queue_position,
            'active_users': active_live_users,
            'max_users': MAX_LIVE_USERS
        })
        return
    
    # Allow live detection
    active_sessions[session_id]['live_detection_active'] = True
    if session_id in live_detection_queue:
        live_detection_queue.remove(session_id)
        active_sessions[session_id]['in_queue'] = False
    
    emit('live_detection_response', {
        'allowed': True,
        'active_users': active_live_users + 1,
        'max_users': MAX_LIVE_USERS
    })

@socketio.on('stop_live_detection')
def handle_stop_live_detection():
    """Handle request to stop live detection"""
    session_id = request.sid
    
    if session_id in active_sessions:
        active_sessions[session_id]['live_detection_active'] = False
        
        # Process queue - allow next person in line
        if live_detection_queue:
            next_session_id = live_detection_queue.pop(0)
            if next_session_id in active_sessions:
                active_sessions[next_session_id]['live_detection_active'] = True
                active_sessions[next_session_id]['in_queue'] = False
                socketio.emit('live_detection_available', {
                    'allowed': True,
                    'message': 'You can now start live detection!'
                }, room=next_session_id)

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    session_id = request.sid
    if session_id in active_sessions:
        session = active_sessions[session_id]
        session_duration = time.time() - session['start_time']
        print(f"Client disconnected: {session_id}, Duration: {session_duration:.2f}s, Confirmed leaks: {len(session['confirmed_leaks'])}")
        
        # Remove from queue if present
        if session_id in live_detection_queue:
            live_detection_queue.remove(session_id)
        
        # If this user was doing live detection, process queue
        if session.get('live_detection_active', False):
            if live_detection_queue:
                next_session_id = live_detection_queue.pop(0)
                if next_session_id in active_sessions:
                    active_sessions[next_session_id]['live_detection_active'] = True
                    active_sessions[next_session_id]['in_queue'] = False
                    socketio.emit('live_detection_available', {
                        'allowed': True,
                        'message': 'You can now start live detection!'
                    }, room=next_session_id)
        
        del active_sessions[session_id]

@socketio.on('process_frame')
def handle_frame(data):
    """Handle frame processing request"""
    session_id = request.sid
    frame_data = data.get('frame')
    
    if not frame_data:
        emit('error', {'message': 'No frame data received'})
        return
    
    # Check if user is allowed to do live detection
    if session_id not in active_sessions or not active_sessions[session_id].get('live_detection_active', False):
        emit('error', {'message': 'Live detection not authorized'})
        return
    
    result = process_frame(frame_data, session_id)
    
    if result:
        emit('detection_result', result)
    else:
        emit('error', {'message': 'Failed to process frame'})

@socketio.on('get_session_stats')
def handle_get_stats():
    """Get current session statistics"""
    session_id = request.sid
    if session_id in active_sessions:
        session = active_sessions[session_id]
        stats = {
            'session_duration': time.time() - session['start_time'],
            'leak_frames': session['leak_frames'],
            'confirmed_leaks': len(session['confirmed_leaks']),
            'total_detections': len(session['leak_detections'])
        }
        emit('session_stats', stats)

@socketio.on('get_queue_status')
def handle_get_queue_status():
    """Get current queue status"""
    session_id = request.sid
    active_live_users = len([s for s in active_sessions.values() if s.get('live_detection_active', False)])
    
    if session_id in active_sessions:
        session = active_sessions[session_id]
        queue_position = None
        if session.get('in_queue', False) and session_id in live_detection_queue:
            queue_position = live_detection_queue.index(session_id) + 1
        
        emit('queue_status', {
            'active_users': active_live_users,
            'max_users': MAX_LIVE_USERS,
            'queue_length': len(live_detection_queue),
            'queue_position': queue_position,
            'live_detection_active': session.get('live_detection_active', False),
            'in_queue': session.get('in_queue', False)
        })

if __name__ == '__main__':
    print("Starting leak detection web application...")
    
    # Initialize model and database
    compiled_model = load_openvino_model()
    client, collection = setup_mongodb()
    
    if compiled_model is None:
        print("Warning: Could not load OpenVINO model. Detection will not work.")
    
    print("Starting Flask-SocketIO server on http://localhost:5002")
    socketio.run(app, host='0.0.0.0', port=5002, debug=True)
