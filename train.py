import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture


def calculate_weights(df):
    pesos_crimen = {
        'Homicidio': 10.0,
        'Extorsión': 8.0,
        'Robo': 5.0,
        'Hurto': 2.0,
        'Estafa': 1.5,
        'Violencia contra la mujer e integrantes': 3.0,
        'Otros': 1.0
    }

    df['peso_crimen'] = df['P_MODALIDADES'].map(pesos_crimen).fillna(1.0)
    anio_min = df['ANIO'].min()
    df['peso_temporal'] = 1 + (df['ANIO'] - anio_min) * 0.2
    df['cantidad_weighted'] = (
        df['cantidad'] * df['peso_crimen'] * df['peso_temporal']
    )
    return df


def sort_clusters(model, X_scaled, clusters):
    cluster_means = []
    for i in range(model.n_components):
        cluster_means.append(X_scaled[clusters == i].mean())

    new_map = {
        old: new
        for new, old in enumerate(np.argsort(cluster_means))
    }
    return np.array([new_map[c] for c in clusters])


def find_best_k_gmm(X, k_range, gmm_kwargs=None):
    bics, models = [], []
    gmm_kwargs = gmm_kwargs or {}

    for k in k_range:
        gmm = GaussianMixture(n_components=k, **gmm_kwargs)
        gmm.fit(X)
        bics.append(gmm.bic(X))
        models.append(gmm)

    best_idx = np.argmin(bics)
    return k_range[best_idx], models[best_idx]


def train_logic_gmm(df_input, k_range, prefix, gmm_kwargs=None):
    print(f"Entrenando {prefix}...")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_input.values)

    best_k, best_model = find_best_k_gmm(X_scaled, k_range, gmm_kwargs=gmm_kwargs)
    print(f"K óptimo {prefix}: {best_k}")

    clusters = best_model.predict(X_scaled)
    clusters = sort_clusters(best_model, X_scaled, clusters)

    joblib.dump(best_model, f"modelo_gmm_{prefix}.joblib")
    joblib.dump(scaler, f"scaler_{prefix}.joblib")

    return clusters


def run_training(k_min: int = 2, k_max: int = 11, covariance_type: str = "diag",
                 max_iter: int = 500, n_init: int = 10, random_state: int = 42):
    df = pd.read_csv("DATASET_Denuncias_Policiales_Ene 2018 a Nov 2025.csv")

    df = calculate_weights(df)

    # ===============================
    # PIVOT DISTRITAL
    # ===============================
    df_pivot_weighted = df.pivot_table(
        index=["UBIGEO_HECHO", "DPTO_HECHO_NEW", "PROV_HECHO", "DIST_HECHO"],
        columns="P_MODALIDADES",
        values="cantidad_weighted",
        aggfunc="sum"
    ).fillna(0)

    df_pivot_real = df.pivot_table(
        index=["UBIGEO_HECHO", "DPTO_HECHO_NEW", "PROV_HECHO", "DIST_HECHO"],
        columns="P_MODALIDADES",
        values="cantidad",
        aggfunc="sum"
    ).fillna(0)

    # ===============================
    # UNIFICAR DEPARTAMENTOS
    # ===============================
    def unificar_departamentos(nombre):
        nombre = str(nombre).upper()
        if "LIMA" in nombre:
            return "LIMA"
        if "CALLAO" in nombre:
            return "CALLAO"
        return nombre

    # --- WEIGHTED ---
    df_dept_weighted = df_pivot_weighted.reset_index()
    df_dept_weighted["DPTO_UNIFICADO"] = (
        df_dept_weighted["DPTO_HECHO_NEW"]
        .apply(unificar_departamentos)
    )

    df_departamental_weighted = (
        df_dept_weighted
        .groupby("DPTO_UNIFICADO")
        .sum(numeric_only=True)
    )

    # --- REAL ---
    df_dept_real = df_pivot_real.reset_index()
    df_dept_real["DPTO_UNIFICADO"] = (
        df_dept_real["DPTO_HECHO_NEW"]
        .apply(unificar_departamentos)
    )

    df_departamental_real = (
        df_dept_real
        .groupby("DPTO_UNIFICADO")
        .sum(numeric_only=True)
    )

    # ===============================
    # ENTRENAMIENTO
    # ===============================
    k_range = range(k_min, k_max + 1)
    gmm_kwargs = {
        "covariance_type": covariance_type,
        "max_iter": max_iter,
        "n_init": n_init,
        "random_state": random_state,
    }

    clusters_dept = train_logic_gmm(
        df_departamental_weighted,
        k_range,
        "departamental",
        gmm_kwargs=gmm_kwargs
    )

    clusters_dist = train_logic_gmm(
        df_pivot_weighted,
        k_range,
        "distrital",
        gmm_kwargs=gmm_kwargs
    )

    # ===============================
    # ASIGNAR CLÚSTERES (MISMO ORDEN)
    # ===============================
    df_departamental_real["cluster_danger_level"] = clusters_dept
    df_pivot_real["cluster_danger_level"] = clusters_dist

    # ===============================
    # GUARDAR RESULTADOS
    # ===============================
    df_departamental_real.reset_index().to_csv(
        "resultados_departamentales.csv",
        index=False
    )

    df_pivot_real.reset_index().to_csv(
        "resultados_distritales.csv",
        index=False
    )


if __name__ == "__main__":
    run_training()
