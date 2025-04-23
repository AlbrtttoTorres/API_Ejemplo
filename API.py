import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from flask import Flask, request, jsonify
import pickle
import numpy as np
import os

app = Flask(__name__)

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/")
def index():
    return jsonify({
        " Bienvenido a la API de predicción de libros ": "Esta API predice la puntuación media (average_rating) de un libro a partir de sus características.",
        " ¿Qué puedes hacer?": {
            " GET /predict?data=ratings_count,text_reviews_count,num_pages":
                "Usa esta ruta para enviar los datos en la URL y obtener una predicción.",
            " POST /predict_json": 
                "Envía un JSON con los datos para obtener la predicción. Formato: {'data': [ratings_count, text_reviews_count, num_pages]}",
        },
        " Formato esperado para los datos": "[ratings_count, text_reviews_count, num_pages]",
        " Ejemplo de uso GET": "/predict?data=5000,800,320",
    })

@app.route("/predict")
def predict():
    data_str = request.args.get("data")
    if not data_str:
        return jsonify({"error": "Proporciona datos separados por comas"}), 400
    try:
        values = np.array([float(x) for x in data_str.split(",")]).reshape(1, -1)
        prediction = model.predict(values)
        return jsonify({"prediction": round(prediction.tolist()[0],2)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/predict_json", methods=["POST"])
def predict_json():
    try:
        input_data = request.get_json()
        values = np.array(input_data["data"]).reshape(1, -1)
        prediction = model.predict(values)
        return jsonify({"prediction": prediction.tolist()[0]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route("/retrain", methods=["POST"])
def retrain():
    try:
        
        if "file" not in request.files:
            return jsonify({"error": "No se encontró el archivo"}), 400
        
        file = request.files["file"]
        
        df = pd.read_csv(file)
        
        X = df[["ratings_count", "text_reviews_count", "num_pages"]]
        y = df["average_rating"]
        
        new_model = RandomForestRegressor()
        new_model.fit(X, y)
        
        with open("model.pkl", "wb") as f:
            pickle.dump(new_model, f)
        
        global model
        model = new_model
        
        return jsonify({"mensaje": "Modelo reentrenado correctamente"})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug_mode = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug_mode)
