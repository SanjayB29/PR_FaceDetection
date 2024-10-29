import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FaceRecognitionModelTrainer:
    """
    A class to handle model selection, training, and evaluation for face recognition.
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the model trainer with available classifiers and their parameter grids.
        
        Args:
            random_state (int): Seed for reproducibility
        """
        self.random_state = random_state
        
        # Initialize classifiers with their default parameters
        self.classifiers = {
            'SVM': {
                'model': SVC(random_state=random_state),
                'params': {
                    'C': [0.1, 1, 10, 100],
                    'kernel': ['rbf', 'linear'],
                    'gamma': ['scale', 'auto', 0.1, 0.01]
                },
                'description': """
                SVM is chosen for its effectiveness in high-dimensional spaces,
                making it suitable for face recognition where features are often
                high-dimensional. It's particularly effective when the number of
                dimensions is greater than the number of samples.
                """
            },
            'RandomForest': {
                'model': RandomForestClassifier(random_state=random_state),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10]
                },
                'description': """
                Random Forest is selected for its robustness to overfitting and
                ability to handle non-linear features. It can capture complex
                patterns in facial features and is less sensitive to noise.
                """
            },
            'KNN': {
                'model': KNeighborsClassifier(),
                'params': {
                    'n_neighbors': [3, 5, 7, 9],
                    'weights': ['uniform', 'distance'],
                    'metric': ['euclidean', 'manhattan']
                },
                'description': """
                KNN is included for its simplicity and effectiveness in pattern
                recognition tasks. It's particularly useful when the relationship
                between features is non-linear and the decision boundary is irregular.
                """
            }
        }
        
        self.scaler = StandardScaler()
        self.best_models = {}
        
    def preprocess_data(self, X: np.ndarray) -> np.ndarray:
        """
        Preprocess the input data by applying standardization.
        
        Args:
            X (np.ndarray): Input features
            
        Returns:
            np.ndarray: Standardized features
        """
        return self.scaler.fit_transform(X)
    
    def train_and_evaluate(self, X: np.ndarray, y: np.ndarray, 
                          test_size: float = 0.2) -> Dict:
        """
        Train and evaluate all models using cross-validation and grid search.
        
        Args:
            X (np.ndarray): Input features
            y (np.ndarray): Target labels
            test_size (float): Proportion of dataset to use for testing
            
        Returns:
            Dict: Dictionary containing training results for all models
        """
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )
        
        # Preprocess the data
        X_train_scaled = self.preprocess_data(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        results = {}
        
        # Train and evaluate each classifier
        for name, clf_info in self.classifiers.items():
            logger.info(f"Training {name}...")
            
            # Perform grid search
            grid_search = GridSearchCV(
                clf_info['model'],
                clf_info['params'],
                cv=5,
                n_jobs=-1,
                scoring='accuracy'
            )
            
            grid_search.fit(X_train_scaled, y_train)
            
            # Store best model
            self.best_models[name] = grid_search.best_estimator_
            
            # Make predictions
            y_pred = grid_search.predict(X_test_scaled)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_test, y_pred, average='weighted'
            )
            conf_matrix = confusion_matrix(y_test, y_pred)
            
            results[name] = {
                'best_params': grid_search.best_params_,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'confusion_matrix': conf_matrix,
                'cross_val_scores': cross_val_score(
                    grid_search.best_estimator_,
                    X_train_scaled,
                    y_train,
                    cv=5
                )
            }
            
            logger.info(f"{name} Results:")
            logger.info(f"Best parameters: {grid_search.best_params_}")
            logger.info(f"Accuracy: {accuracy:.4f}")
            logger.info(f"F1 Score: {f1:.4f}")
            
        return results
    
    def visualize_results(self, results: Dict) -> None:
        """
        Visualize the performance metrics and confusion matrices for all models.
        
        Args:
            results (Dict): Dictionary containing training results
        """
        # Plot performance metrics comparison
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        performance_data = {
            metric: [results[model][metric] for model in results]
            for metric in metrics
        }
        
        plt.figure(figsize=(12, 6))
        df_performance = pd.DataFrame(
            performance_data,
            index=results.keys()
        )
        df_performance.plot(kind='bar', width=0.8)
        plt.title('Model Performance Comparison')
        plt.xlabel('Models')
        plt.ylabel('Score')
        plt.legend(title='Metrics')
        plt.tight_layout()
        plt.show()
        
        # Plot confusion matrices
        n_models = len(results)
        fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 4))
        
        for i, (name, result) in enumerate(results.items()):
            sns.heatmap(
                result['confusion_matrix'],
                annot=True,
                fmt='d',
                ax=axes[i] if n_models > 1 else axes
            )
            axes[i].set_title(f'{name} Confusion Matrix') if n_models > 1 else axes.set_title(f'{name} Confusion Matrix')
        
        plt.tight_layout()
        plt.show()
        
        # Plot cross-validation score distributions
        plt.figure(figsize=(10, 6))
        cross_val_data = [
            results[model]['cross_val_scores'] for model in results
        ]
        plt.boxplot(cross_val_data, labels=results.keys())
        plt.title('Cross-validation Score Distribution')
        plt.xlabel('Models')
        plt.ylabel('Accuracy')
        plt.show()

# Example usage
if __name__ == "__main__":
    # Generate sample data for demonstration
    np.random.seed(42)
    X = np.random.randn(1000, 128)  # 128 facial features
    y = np.random.randint(0, 10, 1000)  # 10 different individuals
    
    # Initialize and run the trainer
    trainer = FaceRecognitionModelTrainer()
    results = trainer.train_and_evaluate(X, y)
    trainer.visualize_results(results)