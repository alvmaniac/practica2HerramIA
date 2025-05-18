from flask import Flask, request, jsonify
import joblib
import mlflow.sklearn
import pandas as pd

mlflow.set_tracking_uri("http://localhost:9090")
app = Flask(__name__)


# Leer el run_id guardado en el archivo
with open("latest_run_id.txt", "r") as f:
    run_id = f.read().strip()

# Cargar Ruta del modelo
model_uri = f"runs:/{run_id}/random_forest_model"
model = mlflow.sklearn.load_model(model_uri)

# Cargar label encoders
label_encoders = joblib.load("label_encoders.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    input_data = request.json
    input_df = pd.DataFrame([input_data])

    for column, encoder in label_encoders.items():
        if column in input_df:
            input_df[column] = encoder.transform(input_df[column])

    prediction = model.predict(input_df)[0]
    prediction_label = list(label_encoders["Avenue"].inverse_transform([prediction]))[0]
    
    return jsonify({"prediction": prediction_label})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9091)
