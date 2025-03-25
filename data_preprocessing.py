import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

def load_data(file_path):
    """Load and clean the dataset"""
    df = pd.read_csv(file_path)
    
    # Data cleaning
    df_clean = df.dropna()
    df.drop_duplicates(inplace=True)
    
    return df

def explore_data(df):
    """Generate exploratory visualizations"""
    # Distribution plots
    plt.figure(figsize=(8, 5))
    sns.histplot(df['PSS_score'], kde=True, bins=20)
    plt.title('Distribution of Perceived Stress Scores (PSS)')
    plt.xlabel('PSS Score')
    plt.ylabel('Frequency')
    plt.show()
    
    # Correlation heatmap
    plt.figure(figsize=(12, 10))
    correlation_matrix = df.corr()
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm')
    plt.title('Correlation Heatmap')
    plt.show()
    
    # Outlier detection
    numeric_columns = ['PSS_score', 'sleep_duration', 'call_duration', 'num_calls', 
                      'num_sms', 'screen_on_time', 'skin_conductance', 'accelerometer', 
                      'mobility_radius', 'mobility_distance']
    
    for column in numeric_columns:
        plt.figure(figsize=(8, 4))
        sns.boxplot(x=df[column])
        plt.title(f'Boxplot of {column}')
        plt.show()

def preprocess_data(df):
    """Normalize and prepare data for modeling"""
    scaler = MinMaxScaler()
    features_to_scale = df.drop(columns=['participant_id', 'day', 'PSS_score'])
    scaled_features = scaler.fit_transform(features_to_scale)
    df_scaled = pd.DataFrame(scaled_features, columns=features_to_scale.columns)
    
    return df_scaled, df