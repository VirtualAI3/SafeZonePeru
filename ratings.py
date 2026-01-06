import os
import json
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from supabase import create_client

TZ = ZoneInfo(os.environ.get("APP_TIMEZONE", "America/Lima"))


# Supabase client (lazy)
_supabase = None


def _get_supabase():
    global _supabase
    if _supabase is not None:
        return _supabase

    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")
    if not url or not key:
        raise RuntimeError("SUPABASE_URL and SUPABASE_KEY must be set in environment")

    _supabase = create_client(url, key)
    return _supabase


# NOTE: Supabase tables (`ratings`, `retrain_logs`) must be provisioned in the
# Supabase project. This module assumes those tables exist with the following
# minimal schema:
# ratings(created_at text, stars int, comment text)
# retrain_logs(started_at text, finished_at text, avg_rating float, low_count int, params_json text, success boolean)


def _now_iso():
    return datetime.now(TZ).isoformat()


def add_rating(stars: int, comment: str | None = None):
    now = _now_iso()
    supabase = _get_supabase()
    payload = {"created_at": now, "stars": int(stars), "comment": comment}
    supabase.table("ratings").insert(payload).execute()


def count_low_ratings(low_star_threshold: int = 2, window_days: int = 7) -> int:
    cutoff = (datetime.now(TZ) - timedelta(days=window_days)).isoformat()
    supabase = _get_supabase()
    res = supabase.table("ratings").select("id", count="exact").lte("stars", low_star_threshold).gte("created_at", cutoff).execute()
    # res.count should be provided by Supabase client; fallback to length
    cnt = res.count if hasattr(res, "count") and res.count is not None else len(res.data or [])
    return int(cnt)


def should_trigger_retrain(low_star_threshold: int = 2, trigger_count: int = 5, window_days: int = 7) -> bool:
    """
    Comprueba si en la ventana `window_days` hay al menos `trigger_count` calificaciones
    con `stars <= low_star_threshold` y además que no se haya hecho un reentrenamiento
    exitoso en esa misma ventana.
    """
    low_cnt = count_low_ratings(low_star_threshold=low_star_threshold, window_days=window_days)
    if low_cnt < trigger_count:
        return False

    # Verificar si ya hubo reentrenamiento exitoso en la ventana
    cutoff = (datetime.now(TZ) - timedelta(days=window_days)).isoformat()
    supabase = _get_supabase()
    res = supabase.table("retrain_logs").select("id", count="exact").gte("started_at", cutoff).eq("success", True).execute()
    recent_retrains = res.count if hasattr(res, "count") and res.count is not None else len(res.data or [])
    return recent_retrains == 0


def _get_rating_stats(window_days: int = 7):
    cutoff = (datetime.now(TZ) - timedelta(days=window_days)).isoformat()
    supabase = _get_supabase()
    res = supabase.table("ratings").select("stars").gte("created_at", cutoff).execute()
    rows = res.data or []
    if not rows:
        return 0.0, 0
    vals = [r.get("stars", 0) for r in rows]
    avg = sum(vals) / len(vals)
    return float(avg), int(len(vals))


def _decide_hyperparams(avg_rating: float, low_count: int) -> dict:
    # Valores base
    params = {
        "k_min": 2,
        "k_max": 11,
        "covariance_type": "diag",
        "max_iter": 500,
        "n_init": 10,
        "random_state": 42,
    }

    # Política simple de 'refuerzo' basada en calificaciones: si las calificaciones
    # medias son bajas, aumentamos ligeramente la exploración de hiperparámetros.
    if avg_rating <= 2.0:
        params["k_max"] += 2
        params["n_init"] += 10
        params["max_iter"] += 300
    elif avg_rating <= 3.0:
        params["k_max"] += 1
        params["n_init"] += 5
        params["max_iter"] += 100

    # Si hay muchos reports bajos, hacer un cambio más agresivo
    if low_count >= 20:
        params["k_max"] += 2
        params["n_init"] += 10

    return params


def _create_retrain_entry(started_at: str, avg_rating: float, low_count: int, params: dict) -> int:
    supabase = _get_supabase()
    payload = {
        "started_at": started_at,
        "avg_rating": avg_rating,
        "low_count": low_count,
        "params_json": json.dumps(params),
        "success": None,
    }
    res = supabase.table("retrain_logs").insert(payload).execute()
    data = res.data or []
    if data:
        return int(data[0].get("id"))
    return 0


def _finalize_retrain_entry(rid: int, finished_at: str, success: bool):
    supabase = _get_supabase()
    supabase.table("retrain_logs").update({"finished_at": finished_at, "success": bool(success)}).eq("id", rid).execute()


def request_retrain(params: dict) -> str:
    """
    Crea un archivo RETRAIN_REQUEST.json en el repositorio con los parámetros
    provistos por la app. Esta función SOLO crea la señal y NO ejecuta
    el entrenamiento ni bloquea la aplicación.

    Devuelve la ruta al archivo creado.
    """
    payload = {
        "requested_at": _now_iso(),
        "params": params,
    }

    # Guardar de forma atómica
    req_path = os.path.join(os.path.dirname(__file__), "RETRAIN_REQUEST.json")
    tmp_path = req_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, req_path)
    return req_path
