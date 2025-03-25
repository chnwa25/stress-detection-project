from data_preprocessing import load_data, explore_data, preprocess_data
from regression_model import build_regression_model, train_regression_model
from classification_model import (categorize_stress, build_classification_model, 
                                 evaluate_classification_model)
from hyperparameter_tuning import build_tuning_model

def main():
    # Load and explore data
    file_path = 'stress_detection.csv'
    df = load_data(file_path)
    explore_data(df)
    
    # Preprocess data
    df_scaled, df = preprocess_data(df)
    
    # Regression model
    X = df_scaled.drop(columns=['PSS_score'])
    y = df_scaled['PSS_score']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    regression_model = build_regression_model(X_train)
    regression_history = train_regression_model(regression_model, X_train, y_train)
    
    # Classification model
    df['Stress_Level'] = df['PSS_score'].apply(categorize_stress)
    X_class = df_scaled.drop(columns=['PSS_score'])
    y_class = df['Stress_Level']
    y_class_encoded = to_categorical(y_class)
    
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
        X_class, y_class_encoded, test_size=0.2, random_state=42)
    
    classification_model = build_classification_model(X_train_c)
    classification_history = classification_model.fit(
        X_train_c, y_train_c, epochs=50, batch_size=32, validation_split=0.2)
    
    evaluate_classification_model(classification_model, X_test_c, y_test_c)
    
    # Save models
    classification_model.save('stress_classification_model.h5')
    regression_model.save('stress_regression_model.h5')

if __name__ == "__main__":
    main()