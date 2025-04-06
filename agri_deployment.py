from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model and label encoders
with open("random_forest_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("label_encoders.pkl", "rb") as encoder_file:
    label_encoders = pickle.load(encoder_file)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    # Preprocess the input data
    for column, encoder in label_encoders.items():
        if column in data:
            data[column] = encoder.transform([data[column]])[0]
    features = np.array([data[col] for col in model.feature_names_in_]).reshape(1, -1)
    prediction = model.predict(features)
    prediction_label = label_encoders["Element"].inverse_transform(prediction)[0]
    return jsonify({"prediction": prediction_label})


if __name__ == "__main__":
    app.run(debug=True)
