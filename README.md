SafeZone - Deploy

Requisitos

- Python 3.10+
- Crear entorno virtual e instalar dependencias:

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

Ejecutar local (Streamlit)

```bash
streamlit run app.py
```

Notas de despliegue

- Para deploy en Streamlit Cloud o similar, sube el repositorio y asegúrate de usar `requirements.txt`.
- `ratings.db` se crea en la raíz del proyecto y contiene tablas `ratings` y `retrain_logs`.
- El reentrenamiento es disparado internamente cuando en la ventana de 7 días se detectan al menos 5 calificaciones <= 2 y no se ha reentrenado en esa semana.
- Los hiperparámetros usados en cada reentrenamiento se almacenan en `retrain_logs` (campo `params_json`).

Consideraciones

- En producción, cambia la política de reentrenamiento por una más robusta (colas, limitación por backoff, notificaciones).
- Si deseas que el reentrenamiento no bloquee la UI, implementa un worker/background (celery, RQ, o subprocess) y registra estado en `retrain_logs`.
