📚 API de Predicción de Average Rating de Libros
=================================================

Esta API predice la valoración promedio (`average_rating`) de un libro a partir de sus características usando un modelo Random Forest.

Desarrollado como práctica grupal final de despliegue para The Bridge.

👥 Equipo de Desarrollo
-------------------------
- Alberto Torres
- Carolina Monzón
- Patricia Galdos
- Enrique Caraballo
- Javier Vidal

⚙️ Tecnologías utilizadas
---------------------------
- Python 3.10+
- Flask
- scikit-learn
- numpy
- pandas

📦 Instalación
----------------
1. Clona el repositorio:
   git clone https://github.com/AlbrtttoTorres/API_Manitas
   cd tu-repo

2. Instala las dependencias:
   pip install -r requirements.txt

3. Ejecuta la API en local:
   python API.py

🚀 Endpoints disponibles
-------------------------

🌐 GET /
---------
Devuelve instrucciones sobre cómo usar la API.

🔍 GET /predict
------------------
Realiza una predicción a partir de tres valores:

Parámetros esperados (en orden):
- ratings_count
- text_reviews_count
- num_pages

Ejemplo:
https://tu-app.onrender.com/predict?data=1000,500,350

Respuesta:
{
  "prediction": 4.23
}

📬 POST /predict_json
------------------------
Permite enviar los datos como JSON:

Cuerpo:
{
  "data": [1000, 500, 350]
}

Respuesta:
{
  "prediction": 4.23
}

🌐 Despliegue
---------------
La API está desplegada en Render:

https://tu-api.onrender.com/

📁 Estructura del Proyecto
---------------------------
📦 tu-repo/
├── API.py              # Código de la API Flask
├── model.py            # Entrenamiento y guardado del modelo
├── model.pkl           # Modelo entrenado
├── requirements.txt    # Dependencias
├── Procfile            # Instrucción de arranque para Render
└── README.md           # Este archivo

✅ Estado del Proyecto
------------------------
✔️ API funcional local y online
✔️ Desplegada en Render
✔️ Modelo Random Forest entrenado con muestra de 1000 libros
✔️ Accesible vía GET y POST
