import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib
import cv2
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

class FaceModelTrainer:
    """
    Class to train and save a Random Forest model for face recognition.
    """
    def __init__(self, data_path=None):
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.model = RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )
        self.label_encoder = LabelEncoder()
        
    def prepare_sample_data(self, n_samples=100, n_classes=5):
        """
        Prepare sample data if no real dataset is provided.
        This is for testing purposes only.
        """
        # Generate synthetic face features
        X = np.random.rand(n_samples, 64*64)  # 64x64 flattened images
        # Generate labels (0 to n_classes-1)
        y = np.random.randint(0, n_classes, n_samples)
        
        return X, y
    
    def train_model(self, X, y):
        """
        Train the Random Forest model.
        """
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42
        )
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Print accuracy
        accuracy = self.model.score(X_test, y_test)
        print(f"Model accuracy: {accuracy:.2f}")
        
        return self.model
    
    def save_model(self, model_path="random_forest_model.joblib"):
        """
        Save the trained model and label encoder.
        """
        # Create a dictionary with both model and label encoder
        model_data = {
            'model': self.model,
            'label_encoder': self.label_encoder
        }
        
        # Save the dictionary
        joblib.dump(model_data, model_path)
        print(f"Model saved to {model_path}")

def main():
    # Create trainer instance
    trainer = FaceModelTrainer()
    
    # Prepare sample data
    X, y = trainer.prepare_sample_data()
    
    # Train model
    model = trainer.train_model(X, y)
    
    # Save model
    trainer.save_model()

if __name__ == "__main__":
    main()