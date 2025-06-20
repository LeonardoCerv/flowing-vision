#!/usr/bin/env python3
"""
Test for the leak detection model with OpenVINO and MongoDB
"""

import cv2
import numpy as np
from openvino import Core

from pymongo import MongoClient
import base64

from datetime import datetime
import time

import os 
from dotenv import load_dotenv
load_dotenv()

"""Setup MongoDB connection"""
def setup_mongodb():
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

"""Load OpenVINO model"""
def load_openvino_model():
    ie = Core()
    model = ie.read_model(model='OpenVinoModel.xml', weights='OpenVinoModel.bin')
    compiled_model = ie.compile_model(model=model, device_name="CPU")

    print("OpenVINO model loaded")
    return compiled_model

"""Initialize camera"""
def initialize_camera():
    cap = cv2.VideoCapture(0, cv2.CAP_ANY)
                
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    print(f"Camera initialized")  
    return cap

"""Convert camera frame to base64 encoded string"""
def encode_frame(frame):
    try:
        is_success, buffer = cv2.imencode(".jpg", frame)
        if not is_success:
            return None
        
        encoded = base64.b64encode(buffer).decode('utf-8')
        return encoded
    
    except Exception as e:
        print(f"Error encoding frame: {str(e)}")
        return None

def main():
    print("Starting leak detection system...")
    
    # Connect to MongoDB
    client, collection = setup_mongodb()
    
    try:
        # Load OpenVINO model
        compiled_model = load_openvino_model()
        
        # Initialize camera
        cap = initialize_camera()
        
        print("All steps successful")
        print("Press 'ESC' to exit")
        

        cv2.namedWindow("Leak Detection", cv2.WINDOW_AUTOSIZE)
        
        leakFrames = 0
        leakDetections = []
        confirmedLeaks = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Can't receive frame. Exiting ...")
                break
            
            h, w, _ = frame.shape
            display_frame = frame.copy()
            
            try:
                # prepare model input
                frameResized = cv2.resize(frame, (640, 640))
                inputData = np.expand_dims(frameResized.transpose(2, 0, 1), axis=0).astype(np.float32) / 255.0
                
                # model inference
                results = compiled_model([inputData])[compiled_model.output(0)]
                
                bestDetection = None
                bestConfidence = 0.5
                
                for result in results[0]: 
                    if result[4] > bestConfidence: #find the latest best detection
                        bestConfidence = result[4]

                        x1, y1, x2, y2 = map(int, result[:4])
                        x1 = int(x1 * w / 640)
                        y1 = int(y1 * h / 640)
                        x2 = int(x2 * w / 640)
                        y2 = int(y2 * h / 640)

                        # save the latest best detection
                        bestDetection = {
                                'confidence': result[4],
                                'bbox': (x1, y1, x2, y2),
                                'timestamp': time.time(),
                                'frame': frame.copy()
                            }

                # process only the best detection (if any)
                if bestDetection:
                    confidence = bestDetection['confidence']
                    x1, y1, x2, y2 = bestDetection['bbox']
                    
                    # draw box of the leak in the video and add a label
                    label = f"leak: {confidence:.2f}"
                    cv2.putText(display_frame, label, (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2) #(0, 165, 255)
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

                    # number of frames a leak has appeared for + compilation of this leak
                    leakFrames += 1
                    leakDetections.append(bestDetection)
                            
                    # Check if we've reached 20 consecutive frames with a leak (confirmed leak)
                    if leakFrames >= 20:
                        # Get the SINGLE best detection (highest confidence) from the compilation
                        bestLeak = max(leakDetections, key=lambda d: d['confidence'])
                        
                        # Store this significant leak for end-of-session saving
                        leakRecord = {
                            'timestamp': datetime.fromtimestamp(bestLeak['timestamp']),
                            'accuracy': float(bestLeak['confidence']),
                            'frames_detected': leakFrames,
                            'bbox': bestLeak['bbox'],
                            'screenshot': encode_frame(bestLeak['frame']),
                            'leak_id' : f"leak_{int(bestLeak['timestamp'])}",
                        }

                        if leakFrames > 20: # make sure to only save one record per leak
                            confirmedLeaks.pop()
                        confirmedLeaks.append(leakRecord)
                        
                        print(f"Significant leak detected for {leakFrames} consecutive frames, confidence: {bestLeak['confidence']:.2f}")
                else:
                    # Reset when no confirmed leak is detected (20+ frames)
                    if leakFrames >= 20:
                        print(f"Leak detection interrupted after {leakFrames} frames")
                    leakFrames = 0
                    leakDetections = []
                
            except Exception as e:
                print(f"Error during inference: {str(e)}")
                break
                
            # show current frame count when tracking a potential leak
            if leakFrames > 0:
                cv2.putText(display_frame, f"Continuous leak: {leakFrames} frames", (10, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # show total confirmed leaks detected in the session
            cv2.putText(display_frame, f"Total leaks detected: {len(confirmedLeaks)}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            cv2.putText(display_frame, "Press ESC to exit", (10, h - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                       
            # Database connection status
            if collection is not None:
                cv2.putText(display_frame, "DB: Connected", (w - 150, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            else:
                cv2.putText(display_frame, "DB: Disconnected", (w - 150, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
            # show the frame with the model's predictions
            cv2.imshow("Leak Detection", display_frame)
  
            # Exit on ESC key
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
        
        print(f"\nSession completed:")

        if collection is not None:
            # Save all confirmed leaks to mongo
            for leak in confirmedLeaks:
                try:
                    collection.insert_one(leak)
                    print(f"Leak {leak['leak_id']} saved to database")
                except Exception as e:
                    print(f"Error saving leak {leak['leak_id']} to database: {str(e)}")

        print(f"Reported leaks: {len(confirmedLeaks)}")

    except Exception as e:
        print(f"Error: {str(e)}")
    
    finally:
        # resource cleanup
        if client is not None:
            client.close()

        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        
        print("Leak detection system stopped")

if __name__ == "__main__":
    main()
