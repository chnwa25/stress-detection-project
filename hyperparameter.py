import keras_tuner as kt
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

def build_tuning_model(hp):
    """Build model for hyperparameter tuning"""
    model = Sequential()
    
    # Tune number of units in first layer
    hp_units = hp.Int('units', min_value=32, max_value=512, step=32)
    model.add(Dense(units=hp_units, activation='relu', input_shape=(X_train_c.shape[1],)))
    
    # Tune dropout rate
    hp_dropout = hp.Float('dropout', min_value=0.0, max_value=0.5, step=0.1)
    model.add(Dropout(rate=hp_dropout))
    
    # Add more layers
    model.add(Dense(units=hp.Int('dense_2_units', 32, 256, 32), activation='relu'))
    model.add(Dense(3, activation='softmax'))
    
    # Tune learning rate
    hp_learning_rate = hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])
    
    model.compile(optimizer=Adam(learning_rate=hp_learning_rate),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model