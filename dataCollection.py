import cv2
import numpy as np
import os
import pandas as pd
from datetime import datetime
import time

class FacialDatasetCapture:
    def __init__(self, root_dir='dataset', max_images=50, use_dlib=False):
        """
        Initialize the facial dataset capture system.
        
        Args:
            root_dir (str): Root directory for storing dataset
            max_images (int): Maximum number of images to capture per subject
            use_dlib (bool): Whether to use Dlib for facial landmarks (optional)
        """
        # Initialize directories and files
        self.root_dir = root_dir
        self.max_images = max_images
        self.csv_path = os.path.join(root_dir, 'data_log.csv')
        self.use_dlib = use_dlib
        
        # Create root directory if it doesn't exist
        if not os.path.exists(root_dir):
            os.makedirs(root_dir)
            
        # Initialize or load the data log
        if os.path.exists(self.csv_path):
            self.data_log = pd.read_csv(self.csv_path)
        else:
            self.data_log = pd.DataFrame(columns=['filepath', 'subject', 'timestamp', 'augmentation'])
            
        # Initialize face detection model
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Initialize Dlib if requested
        if self.use_dlib:
            try:
                import dlib
                self.face_detector = dlib.get_frontal_face_detector()
                self.shape_predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
            except (ImportError, RuntimeError) as e:
                print("Warning: Dlib initialization failed. Falling back to basic face detection.")
                print("Error:", str(e))
                self.use_dlib = False
        
        # Image processing parameters
        self.target_size = (92, 112)
        
    def create_subject_directory(self, subject_name):
        """Create a directory for the subject if it doesn't exist."""
        subject_dir = os.path.join(self.root_dir, subject_name)
        if not os.path.exists(subject_dir):
            os.makedirs(subject_dir)
        return subject_dir
        
    def preprocess_image(self, image, face_rect):
        """
        Preprocess the detected face region.
        
        Args:
            image: Input image
            face_rect: Rectangle coordinates of detected face
        Returns:
            Preprocessed face image
        """
        x, y, w, h = face_rect
        
        # Add padding to the face region (10% on each side)
        padding_x = int(w * 0.1)
        padding_y = int(h * 0.1)
        
        # Calculate padded coordinates
        x1 = max(x - padding_x, 0)
        y1 = max(y - padding_y, 0)
        x2 = min(x + w + padding_x, image.shape[1])
        y2 = min(y + h + padding_y, image.shape[0])
        
        # Extract face region with padding
        face_img = image[y1:y2, x1:x2]
        
        # Convert to grayscale
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        
        # Resize to standard size
        resized = cv2.resize(gray, self.target_size)
        
        # Normalize pixel values
        normalized = resized.astype(np.float32) / 255.0
        
        return normalized
        
    def apply_augmentations(self, image):
        """
        Apply various augmentations to the input image.
        
        Args:
            image: Input image
        Returns:
            List of augmented images with their labels
        """
        augmented = []
        
        # Original image
        augmented.append(('original', image))
        
        # Horizontal flip
        flipped = cv2.flip(image, 1)
        augmented.append(('flip', flipped))
        
        # Rotations
        for angle in [-5, 5]:
            matrix = cv2.getRotationMatrix2D(
                (image.shape[1] / 2, image.shape[0] / 2), angle, 1.0
            )
            rotated = cv2.warpAffine(image, matrix, (image.shape[1], image.shape[0]))
            augmented.append((f'rotate_{angle}', rotated))
            
        # Brightness adjustments
        for alpha in [0.8, 1.2]:
            brightened = cv2.convertScaleAbs(image, alpha=alpha, beta=0)
            augmented.append((f'brightness_{alpha}', brightened))
            
        return augmented
        
    def log_image(self, filepath, subject, augmentation):
        """Log the captured image information to CSV."""
        new_row = pd.DataFrame({
            'filepath': [filepath],
            'subject': [subject],
            'timestamp': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
            'augmentation': [augmentation]
        })
        self.data_log = pd.concat([self.data_log, new_row], ignore_index=True)
        self.data_log.to_csv(self.csv_path, index=False)
        
    def capture_images(self, subject_name):
        """
        Capture and process images for a subject.
        
        Args:
            subject_name: Name or identifier of the subject
        """
        subject_dir = self.create_subject_directory(subject_name)
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
            
        image_count = 0
        
        print(f"\nStarting capture for subject: {subject_name}")
        print(f"Press 'SPACE' to capture or 'Q' to quit")
        print(f"Images remaining: {self.max_images - image_count}")
        
        while image_count < self.max_images:
            ret, frame = cap.read()
            if not ret:
                print("Error reading from webcam")
                break
                
            # Detect faces
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )
            
            # Draw face rectangles
            display_frame = frame.copy()
            for (x, y, w, h) in faces:
                cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
            # Add text overlay
            cv2.putText(display_frame, f"Images: {image_count}/{self.max_images}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
            # Display frame
            cv2.imshow('Face Capture', display_frame)
            key = cv2.waitKey(1) & 0xFF
            
            if key == 32 and len(faces) > 0:  # Spacebar pressed and face detected
                face_rect = faces[0]  # Process the first detected face
                processed_face = self.preprocess_image(frame, face_rect)
                
                # Generate and save augmented images
                augmented_faces = self.apply_augmentations(processed_face)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                for aug_type, aug_face in augmented_faces:
                    if image_count >= self.max_images:
                        break
                        
                    filename = f"{timestamp}_{aug_type}.png"
                    filepath = os.path.join(subject_dir, filename)
                    
                    # Save image
                    cv2.imwrite(filepath, (aug_face * 255).astype(np.uint8))
                    
                    # Log to CSV
                    self.log_image(filepath, subject_name, aug_type)
                    
                    image_count += 1
                    
                print(f"Images remaining: {self.max_images - image_count}")
                time.sleep(0.5)  # Prevent multiple captures
                
            elif key == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()
        print(f"Capture complete for subject: {subject_name}")
        print(f"Total images captured: {image_count}")

def main():
    """Main function to run the facial dataset capture program."""
    print("Facial Dataset Capture Program")
    print("-----------------------------")
    
    # Get configuration from user
    while True:
        try:
            max_images = int(input("Enter maximum number of images per subject: "))
            if max_images > 0:
                break
            print("Please enter a positive number")
        except ValueError:
            print("Please enter a valid number")
    
    capturer = FacialDatasetCapture(max_images=max_images)
    
    while True:
        subject_name = input("\nEnter subject name (or 'quit' to exit): ")
        if subject_name.lower() == 'quit':
            break
            
        capturer.capture_images(subject_name)
        
        continue_capture = input("Capture another subject? (y/n): ")
        if continue_capture.lower() != 'y':
            break
    
    print("\nProgram terminated")
    print(f"Dataset saved in: {capturer.root_dir}")
    print(f"Log file saved as: {capturer.csv_path}")

if __name__ == "__main__":
    main()