# API de predicción con FastAPI

## Endpoints

- `/` — Landing page informativa
- `/predict?x1=...&x2=...` — Devuelve la predicción del modelo
- `/hello` — Endpoint oculto para redespliegue (comentado)

## Cómo correr localmente

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
