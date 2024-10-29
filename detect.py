import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
import threading
import queue
import joblib
from PIL import Image, ImageTk
import logging
from typing import Tuple, Optional
import dlib
import time
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FaceRecognitionApp:
    """
    Live face recognition application using OpenCV and Random Forest classifier.
    """
    
    def __init__(self):
        """Initialize the face recognition application."""
        try:
            # Load the model if it exists, otherwise train a new one
            if os.path.exists("random_forest_model.joblib"):
                model_data = joblib.load("random_forest_model.joblib")
                self.model = model_data['model']
                self.label_encoder = model_data['label_encoder']
            else:
                self.train_model()

            # Initialize face cascade classifier
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            
            # Initialize dlib's face detector
            self.dlib_detector = dlib.get_frontal_face_detector()
            
            # Initialize video capture
            self.cap = None
            self.is_running = False
            
            # Frame processing queue
            self.frame_queue = queue.Queue(maxsize=2)
            
            # Setup GUI
            self.setup_gui()
            
            # Performance tracking
            self.fps = 0
            self.frame_count = 0
            self.start_time = time.time()
            
        except Exception as e:
            logger.error(f"Initialization error: {str(e)}")
            raise

    def train_model(self):
        """Train a new model with sample data."""
        # Generate sample data
        n_samples = 100
        n_features = 64 * 64  # 64x64 image size
        X = np.random.rand(n_samples, n_features)
        y = np.random.randint(0, 5, n_samples)  # 5 different classes

        # Initialize and train the model
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import LabelEncoder

        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.label_encoder = LabelEncoder()
        
        y_encoded = self.label_encoder.fit_transform(y)
        self.model.fit(X, y_encoded)

        # Save the model
        model_data = {
            'model': self.model,
            'label_encoder': self.label_encoder
        }
        joblib.dump(model_data, "random_forest_model.joblib")
        
    def setup_gui(self):
        """Set up the Tkinter GUI."""
        self.root = tk.Tk()
        self.root.title("Face Recognition System")
        
        # Configure main window
        self.root.minsize(800, 600)
        self.root.configure(bg='#f0f0f0')
        
        # Main frame
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Style configuration
        style = ttk.Style()
        style.configure('Control.TFrame', background='#e0e0e0', relief='raised')
        
        # Video frame
        self.video_label = ttk.Label(self.main_frame)
        self.video_label.grid(row=0, column=0, columnspan=3, pady=10)
        
        # Control panel
        control_frame = ttk.Frame(self.main_frame, style='Control.TFrame')
        control_frame.grid(row=1, column=0, columnspan=3, pady=5, sticky='ew')
        
        # Control buttons
        self.start_button = ttk.Button(
            control_frame, 
            text="Start Camera", 
            command=self.start_video,
            width=20
        )
        self.start_button.grid(row=0, column=0, padx=5, pady=5)
        
        self.stop_button = ttk.Button(
            control_frame, 
            text="Stop Camera", 
            command=self.stop_video,
            state=tk.DISABLED,
            width=20
        )
        self.stop_button.grid(row=0, column=1, padx=5, pady=5)
        
        self.quit_button = ttk.Button(
            control_frame, 
            text="Quit", 
            command=self.cleanup,
            width=20
        )
        self.quit_button.grid(row=0, column=2, padx=5, pady=5)
        
        # Status panel
        status_frame = ttk.Frame(self.main_frame)
        status_frame.grid(row=2, column=0, columnspan=3, pady=5, sticky='ew')
        
        self.fps_label = ttk.Label(status_frame, text="FPS: 0")
        self.fps_label.grid(row=0, column=0, padx=5)
        
        self.status_label = ttk.Label(status_frame, text="Status: Ready")
        self.status_label.grid(row=0, column=1, padx=5)
    
    def preprocess_face(self, face_img: np.ndarray, target_size: Tuple[int, int] = (64, 64)) -> np.ndarray:
        """
        Preprocess the detected face for model input.
        
        Args:
            face_img (np.ndarray): Detected face image
            target_size (tuple): Target size for model input
            
        Returns:
            np.ndarray: Preprocessed face image
        """
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            
            # Resize to target size
            resized = cv2.resize(gray, target_size)
            
            # Normalize pixel values
            normalized = resized / 255.0
            
            # Flatten the image for model input
            flattened = normalized.reshape(1, -1)
            
            return flattened
            
        except Exception as e:
            logger.error(f"Error in preprocessing face: {str(e)}")
            return None
    
    def detect_faces(self, frame: np.ndarray) -> list:
        """
        Detect faces in the frame using both Haar cascade and dlib.
        
        Args:
            frame (np.ndarray): Input frame from video stream
            
        Returns:
            list: List of detected face coordinates
        """
        try:
            # Convert frame to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces using Haar cascade
            faces_haar = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            
            # Convert faces_haar to list if it's a numpy array
            faces_haar = list(faces_haar) if len(faces_haar) > 0 else []
            
            return faces_haar
            
        except Exception as e:
            logger.error(f"Error in face detection: {str(e)}")
            return []
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Process a single frame for face detection and recognition.
        
        Args:
            frame (np.ndarray): Input frame from video stream
            
        Returns:
            np.ndarray: Processed frame with annotations
        """
        try:
            # Detect faces
            faces = self.detect_faces(frame)
            
            for (x, y, w, h) in faces:
                # Extract and preprocess face region
                face_roi = frame[y:y+h, x:x+w]
                processed_face = self.preprocess_face(face_roi)
                
                if processed_face is not None:
                    # Make prediction
                    prediction_encoded = self.model.predict(processed_face)[0]
                    prediction = self.label_encoder.inverse_transform([prediction_encoded])[0]
                    confidence = np.max(self.model.predict_proba(processed_face))
                    
                    # Draw bounding box
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    
                    # Display prediction and confidence
                    label = f"Person {prediction} ({confidence:.2f})"
                    cv2.putText(
                        frame,
                        label,
                        (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (0, 255, 0),
                        2
                    )
            
            # Calculate and display FPS
            self.frame_count += 1
            elapsed_time = time.time() - self.start_time
            if elapsed_time > 1:
                self.fps = self.frame_count / elapsed_time
                self.frame_count = 0
                self.start_time = time.time()
                self.fps_label.config(text=f"FPS: {self.fps:.2f}")
            
            return frame
            
        except Exception as e:
            logger.error(f"Error in frame processing: {str(e)}")
            return frame
    
    def video_loop(self):
        """Main video processing loop."""
        try:
            while self.is_running:
                ret, frame = self.cap.read()
                if not ret:
                    break
                    
                # Process frame
                processed_frame = self.process_frame(frame)
                
                # Convert to PhotoImage for Tkinter
                rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(rgb_frame)
                
                # Resize image to fit display if needed
                display_size = (800, 600)  # Adjust as needed
                img.thumbnail(display_size, Image.Resampling.LANCZOS)
                
                photo = ImageTk.PhotoImage(image=img)
                
                # Update GUI
                self.video_label.config(image=photo)
                self.video_label.image = photo
                
            self.cap.release()
            
        except Exception as e:
            logger.error(f"Error in video loop: {str(e)}")
            self.stop_video()
    
    def start_video(self):
        """Start video capture and processing."""
        try:
            if not self.is_running:
                self.cap = cv2.VideoCapture(0)
                if not self.cap.isOpened():
                    messagebox.showerror("Error", "Could not open camera")
                    return
                    
                self.is_running = True
                self.status_label.config(text="Status: Running")
                
                # Start video processing thread
                self.video_thread = threading.Thread(target=self.video_loop)
                self.video_thread.daemon = True
                self.video_thread.start()
                
                # Update button states
                self.start_button.config(state=tk.DISABLED)
                self.stop_button.config(state=tk.NORMAL)
                
        except Exception as e:
            logger.error(f"Error starting video: {str(e)}")
            messagebox.showerror("Error", str(e))
    
    def stop_video(self):
        """Stop video capture and processing."""
        try:
            self.is_running = False
            if self.cap is not None:
                self.cap.release()
            
            self.status_label.config(text="Status: Stopped")
            
            # Update button states
            self.start_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)
            
        except Exception as e:
            logger.error(f"Error stopping video: {str(e)}")
    
    def cleanup(self):
        """Clean up resources and quit application."""
        try:
            self.stop_video()
            self.root.quit()
            
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")
            
    def run(self):
        """Run the application."""
        try:
            # Set up window close handler
            self.root.protocol("WM_DELETE_WINDOW", self.cleanup)
            
            # Start main loop
            self.root.mainloop()
            
        except Exception as e:
            logger.error(f"Error running application: {str(e)}")
            raise

def main():
    try:
        # Create and run the application
        app = FaceRecognitionApp()
        app.run()
        
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        messagebox.showerror("Error", str(e))

if __name__ == "__main__":
    main()