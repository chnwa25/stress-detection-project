from flask import Flask, request, jsonify
import numpy as np
from keras.models import load_model

# Initialize Flask app
app = Flask(__name__)

# Load your trained model
model = load_model('stress_classification_model.h5')  # or use .h5 if you saved as h5

# Define a route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Expect JSON input
        data = request.get_json(force=True)

        # Example: Extract features from request
        features = data['features']  # Expects a list of feature values

        # Convert features to a NumPy array and reshape for prediction
        input_array = np.array(features).reshape(1, -1)

        # Make prediction (probabilities)
        prediction_prob = model.predict(input_array)

        # Convert probabilities to class index
        predicted_class = int(np.argmax(prediction_prob, axis=1)[0])

        # Map class index to label
        label_map = {0: 'Low Stress', 1: 'Medium Stress', 2: 'High Stress'}
        result = label_map[predicted_class]

        # Return JSON response
        return jsonify({
            'prediction': result,
            'probability': prediction_prob.tolist()
        })

    except Exception as e:
        return jsonify({'error': str(e)})

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)