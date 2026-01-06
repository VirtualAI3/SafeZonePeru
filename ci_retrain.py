import json
import os
import sys
from datetime import datetime

import train


def main():
    repo_root = os.path.dirname(__file__)
    req_path = os.path.join(repo_root, "RETRAIN_REQUEST.json")

    if not os.path.exists(req_path):
        print("CI: RETRAIN_REQUEST.json no encontrado; nothing to do.")
        return 0

    try:
        with open(req_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception as e:
        print(f"CI: error leyendo RETRAIN_REQUEST.json: {e}")
        return 2

    params = payload.get("params", {}) or {}
    print(f"CI: reentrenamiento solicitado en {payload.get('requested_at')} con params: {params}")

    try:
        train.run_training(**params)
        # Optionally write a result log
        result = {
            "retrained_at": datetime.utcnow().isoformat() + "Z",
            "params": params,
            "status": "success",
        }
        with open(os.path.join(repo_root, "RETRAIN_RESULT.json"), "w", encoding="utf-8") as rf:
            json.dump(result, rf, ensure_ascii=False, indent=2)
        print("CI: reentrenamiento completado correctamente.")
        return 0
    except Exception as e:
        print(f"CI: error durante reentrenamiento: {e}")
        result = {
            "retrained_at": datetime.utcnow().isoformat() + "Z",
            "params": params,
            "status": "failure",
            "error": str(e),
        }
        with open(os.path.join(repo_root, "RETRAIN_RESULT.json"), "w", encoding="utf-8") as rf:
            json.dump(result, rf, ensure_ascii=False, indent=2)
        return 3


if __name__ == "__main__":
    code = main()
    sys.exit(code)
