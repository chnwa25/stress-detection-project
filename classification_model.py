import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

def categorize_stress(score):
    """Categorize stress levels"""
    if score <= 20:
        return 0  # Low stress
    elif score <= 30:
        return 1  # Medium stress
    else:
        return 2  # High stress

def build_classification_model(X_train):
    """Build and compile classification model"""
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(32, activation='relu'),
        Dense(3, activation='softmax')
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.001),
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])
    
    return model

def evaluate_classification_model(model, X_test_c, y_test_c):
    """Evaluate and visualize classification results"""
    # Predict probabilities
    y_pred_probs = model.predict(X_test_c)
    y_pred_classes = np.argmax(y_pred_probs, axis=1)
    y_true_classes = np.argmax(y_test_c, axis=1)
    
    # Confusion matrix
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Low', 'Medium', 'High'], 
                yticklabels=['Low', 'Medium', 'High'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()
    
    # Classification report
    report = classification_report(y_true_classes, y_pred_classes, 
                                  target_names=['Low', 'Medium', 'High'])
    print(report)