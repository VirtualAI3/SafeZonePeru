import sqlite3
import os
import json
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

DB_PATH = os.path.join(os.path.dirname(__file__), "database.db")
TZ = ZoneInfo("America/Lima")


def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS ratings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT,
            stars INTEGER,
            comment TEXT
        )
        """
    )
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS retrain_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            started_at TEXT,
            finished_at TEXT,
            avg_rating REAL,
            low_count INTEGER,
            params_json TEXT,
            success INTEGER
        )
        """
    )
    conn.commit()
    conn.close()


def _now_iso():
    return datetime.now(TZ).isoformat()


def add_rating(stars: int, comment: str | None = None):
    init_db()
    now = _now_iso()
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "INSERT INTO ratings (created_at, stars, comment) VALUES (?, ?, ?)",
        (now, int(stars), comment),
    )
    conn.commit()
    conn.close()


def count_low_ratings(low_star_threshold: int = 2, window_days: int = 7) -> int:
    init_db()
    cutoff = (datetime.now(TZ) - timedelta(days=window_days)).isoformat()
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "SELECT COUNT(*) FROM ratings WHERE stars <= ? AND created_at >= ?",
        (low_star_threshold, cutoff),
    )
    cnt = c.fetchone()[0]
    conn.close()
    return int(cnt)


def should_trigger_retrain(low_star_threshold: int = 2, trigger_count: int = 5, window_days: int = 7) -> bool:
    """
    Comprueba si en la ventana `window_days` hay al menos `trigger_count` calificaciones
    con `stars <= low_star_threshold` y además que no se haya hecho un reentrenamiento
    exitoso en esa misma ventana.
    """
    init_db()
    low_cnt = count_low_ratings(low_star_threshold=low_star_threshold, window_days=window_days)
    if low_cnt < trigger_count:
        return False

    # Verificar si ya hubo reentrenamiento exitoso en la ventana
    cutoff = (datetime.now(TZ) - timedelta(days=window_days)).isoformat()
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "SELECT COUNT(*) FROM retrain_logs WHERE started_at >= ? AND success = 1",
        (cutoff,),
    )
    recent_retrains = c.fetchone()[0]
    conn.close()
    return recent_retrains == 0


def _get_rating_stats(window_days: int = 7):
    init_db()
    cutoff = (datetime.now(TZ) - timedelta(days=window_days)).isoformat()
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT AVG(stars), COUNT(*) FROM ratings WHERE created_at >= ?", (cutoff,))
    row = c.fetchone()
    conn.close()
    if row is None:
        return 0.0, 0
    avg = row[0] or 0.0
    cnt = row[1] or 0
    return float(avg), int(cnt)


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
    init_db()
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "INSERT INTO retrain_logs (started_at, avg_rating, low_count, params_json, success) VALUES (?, ?, ?, ?, NULL)",
        (started_at, avg_rating, low_count, json.dumps(params)),
    )
    conn.commit()
    rid = c.lastrowid
    conn.close()
    return int(rid)


def _finalize_retrain_entry(rid: int, finished_at: str, success: bool):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "UPDATE retrain_logs SET finished_at = ?, success = ? WHERE id = ?",
        (finished_at, 1 if success else 0, rid),
    )
    conn.commit()
    conn.close()


def trigger_retrain() -> bool:
    """
    Reentrena internamente usando una política basada en calificaciones (no expuesta
    al usuario). Devuelve True si el reentrenamiento finaliza correctamente.
    """
    init_db()

    window_days = 7
    low_threshold = 2

    # Comprobar condición de disparo (usa ventana semanal)
    if not should_trigger_retrain(low_star_threshold=low_threshold, trigger_count=5, window_days=window_days):
        return False

    # Estadísticas y conteos en la ventana
    avg, total = _get_rating_stats(window_days=window_days)
    low = count_low_ratings(low_star_threshold=low_threshold, window_days=window_days)

    params = _decide_hyperparams(avg_rating=avg, low_count=low)

    started = _now_iso()
    rid = _create_retrain_entry(started_at=started, avg_rating=avg, low_count=low, params=params)

    success = False
    try:
        import train

        train.run_training(**params)
        success = True
    except Exception:
        success = False
    finally:
        finished = _now_iso()
        _finalize_retrain_entry(rid, finished_at=finished, success=success)

    return success


# Inicializa DB al importar
init_db()