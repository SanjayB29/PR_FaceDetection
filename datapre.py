import numpy as np
import pandas as pd
import os
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
import cv2
from typing import Tuple, List, Dict
import logging

class FacialFeatureExtractor:
    def __init__(self, n_components_pca: int = 50, n_components_lda: int = None, use_lda: bool = True):
        """
        Initialize the facial feature extractor.
        
        Args:
            n_components_pca (int): Number of principal components to keep
            n_components_lda (int): Number of LDA components (defaults to n_classes - 1)
            use_lda (bool): Whether to apply LDA after PCA
        """
        self.n_components_pca = n_components_pca
        self.n_components_lda = n_components_lda
        self.use_lda = use_lda
        
        # Initialize transformers
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_components_pca)
        self.lda = LinearDiscriminantAnalysis(n_components=n_components_lda) if use_lda else None
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def load_image(self, filepath: str) -> np.ndarray:
        """
        Load and verify image dimensions.
        
        Args:
            filepath: Path to the image file
            
        Returns:
            Flattened image array
        """
        # Load grayscale image
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        
        # Verify dimensions
        if img.shape != (112, 92):
            raise ValueError(f"Image {filepath} has incorrect dimensions. Expected (112, 92), got {img.shape}")
        
        # Flatten the image to 1D array
        return img.flatten()
        
    def extract_basic_features(self, img_array: np.ndarray) -> Dict[str, float]:
        """
        Extract basic statistical features from image.
        
        Args:
            img_array: Flattened image array
            
        Returns:
            Dictionary of basic features
        """
        return {
            'mean_intensity': np.mean(img_array),
            'std_intensity': np.std(img_array),
            'median_intensity': np.median(img_array),
            'min_intensity': np.min(img_array),
            'max_intensity': np.max(img_array),
            'range_intensity': np.ptp(img_array)
        }
        
    def prepare_dataset(self, data_dir: str, csv_path: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Load images and prepare dataset for feature extraction.
        
        Args:
            data_dir: Root directory containing image folders
            csv_path: Path to the CSV file containing image metadata
            
        Returns:
            Tuple of (image_data, labels, feature_names)
        """
        # Load the CSV file
        df = pd.read_csv(csv_path)
        
        # Initialize lists for data
        image_data = []
        labels = []
        basic_features_list = []
        
        # Process each image
        for _, row in df.iterrows():
            try:
                # Load and flatten image
                img_array = self.load_image(row['filepath'])
                
                # Extract basic features
                basic_features = self.extract_basic_features(img_array)
                
                # Store data
                image_data.append(img_array)
                labels.append(row['subject'])
                basic_features_list.append(basic_features)
                
            except Exception as e:
                self.logger.warning(f"Error processing {row['filepath']}: {str(e)}")
                continue
        
        # Convert to numpy arrays
        X = np.array(image_data)
        y = np.array(labels)
        
        # Create feature names
        feature_names = [f'pixel_{i}' for i in range(X.shape[1])]
        feature_names.extend(basic_features_list[0].keys())
        
        # Combine pixel values with basic features
        basic_features_array = np.array([list(f.values()) for f in basic_features_list])
        X = np.hstack((X, basic_features_array))
        
        return X, y, feature_names
        
    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        """
        Fit the feature extractors and transform the data.
        
        Args:
            X: Input data matrix
            y: Labels
            
        Returns:
            Tuple of (transformed_features, feature_names)
        """
        # Scale the data
        X_scaled = self.scaler.fit_transform(X)
        
        # Apply PCA
        X_pca = self.pca.fit_transform(X_scaled)
        
        # Generate PCA feature names
        feature_names = [f'pca_component_{i}' for i in range(self.n_components_pca)]
        
        # Calculate explained variance ratio
        explained_variance_ratio = self.pca.explained_variance_ratio_
        cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
        self.logger.info(f"Cumulative explained variance ratio: {cumulative_variance_ratio[-1]:.3f}")
        
        # Apply LDA if requested
        if self.use_lda:
            X_transformed = self.lda.fit_transform(X_pca, y)
            feature_names = [f'lda_component_{i}' for i in range(X_transformed.shape[1])]
        else:
            X_transformed = X_pca
        
        return X_transformed, feature_names
        
    def save_features(self, features: np.ndarray, labels: np.ndarray, 
                     feature_names: List[str], output_path: str):
        """
        Save extracted features to CSV file.
        
        Args:
            features: Extracted feature matrix
            labels: Subject labels
            feature_names: Names of features
            output_path: Path to save the CSV file
        """
        # Create DataFrame
        df = pd.DataFrame(features, columns=feature_names)
        df['subject'] = labels
        
        # Save to CSV
        df.to_csv(output_path, index=False)
        self.logger.info(f"Features saved to {output_path}")
        
        # Log feature statistics
        self.logger.info(f"Dataset shape: {df.shape}")
        self.logger.info(f"Number of subjects: {len(df['subject'].unique())}")
        
def main():
    """Main function to demonstrate feature extraction pipeline."""
    # Configuration
    data_dir = 'dataset'  # Root directory containing image folders
    csv_path = os.path.join(data_dir, 'data_log.csv')  # Path to metadata CSV
    output_path = os.path.join(data_dir, 'facial_features.csv')  # Output path
    
    # Initialize feature extractor
    extractor = FacialFeatureExtractor(
        n_components_pca=50,  # Retain top 50 principal components
        use_lda=True  # Apply LDA after PCA
    )
    
    # Load and prepare dataset
    X, y, initial_feature_names = extractor.prepare_dataset(data_dir, csv_path)
    
    # Extract features
    features, feature_names = extractor.fit_transform(X, y)
    
    # Save features
    extractor.save_features(features, y, feature_names, output_path)
    
if __name__ == "__main__":
    main()