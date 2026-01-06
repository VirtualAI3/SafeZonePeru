import streamlit as st
import pandas as pd
import plotly.express as px
import json
import ratings

st.set_page_config(
    page_title="SafeZone Peru - Inteligencia Criminal GMM",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilo visual
st.markdown("""
<style>
.main { background-color: #fafafa; }
.stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data(file_path, nivel):
    df = pd.read_csv(file_path)
    if nivel == "Distrital":
        df["ID_MAPA"] = df["UBIGEO_HECHO"].astype(int).astype(str).str.zfill(6)
    else:
        df["ID_MAPA"] = df["DPTO_UNIFICADO"].astype(str).str.upper()
    return df

@st.cache_data
def load_geojson(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def clean_geojson(geojson):
    geojson["features"] = [
        f for f in geojson["features"]
        if f.get("geometry") and f["geometry"].get("coordinates")
    ]
    return geojson

st.sidebar.title("Configuración SafeZone")
nivel_analisis = st.sidebar.radio(
    "Seleccione Nivel de Análisis:",
    ["Distrital", "Departamental"]
)

if nivel_analisis == "Distrital":
    csv_path = "resultados_distritales.csv"
    geojson_path = "peru_distrital_simple.geojson"
    geo_key_plotly = "properties.IDDIST"
    df_col_nombre = "DIST_HECHO"
else:
    csv_path = "resultados_departamentales.csv"
    geojson_path = "peru_departamental_simple.geojson"
    geo_key_plotly = "properties.NOMBDEP"
    df_col_nombre = "DPTO_UNIFICADO"

try:
    df_raw = load_data(csv_path, nivel_analisis)
    geojson_data = clean_geojson(load_geojson(geojson_path))
except FileNotFoundError as e:
    st.error(f"Faltan archivos de datos: {e}")
    st.stop()

n_clusters = df_raw["cluster_danger_level"].nunique()
df_raw["status"] = df_raw["cluster_danger_level"].apply(lambda x: f"Nivel {int(x)}")


color_sequence = px.colors.sample_colorscale("Reds", [i/(n_clusters-1) for i in range(n_clusters)])
all_levels = sorted(df_raw["status"].unique(), key=lambda x: int(x.split()[1]))
color_map = {nivel: color for nivel, color in zip(all_levels, color_sequence)}

jurisdicciones_lista = sorted(df_raw[df_col_nombre].dropna().unique())

st.title(f"Mapa de Peligrosidad Criminal - {nivel_analisis}")
st.info(f"Se han identificado {n_clusters} niveles de riesgo mediante el modelo GMM.")

col_map, col_info = st.columns([2.2, 1])

with col_map:
    delitos_cols = [
        "Hurto", "Robo", "Estafa", "Extorsión", 
        "Homicidio", "Otros", "Violencia contra la mujer e integrantes"
    ]
    hover_cols = [c for c in delitos_cols if c in df_raw.columns]

    fig = px.choropleth_mapbox(
        df_raw,
        geojson=geojson_data,
        locations="ID_MAPA" if nivel_analisis == "Distrital" else "DPTO_UNIFICADO",
        featureidkey=geo_key_plotly,
        color="status",
        color_discrete_map=color_map,
        hover_name=df_col_nombre,
        hover_data=hover_cols,
        mapbox_style="carto-positron",
        center={"lat": -9.19, "lon": -75.02},
        zoom=4.5 if nivel_analisis == "Departamental" else 5,
        opacity=0.8,
        category_orders={"status": all_levels}
    )

    fig.update_layout(
        height=750,
        margin=dict(r=0, l=0, t=0, b=0),
        legend_title_text="Escala de Peligro",
        clickmode="event+select"
    )

    selection = st.plotly_chart(
        fig, 
        use_container_width=True, 
        on_select="rerun", 
        key="mapa_peru",
        config={"scrollZoom": True, "displayModeBar": True}
    )

    if selection and selection.get("selection") and selection["selection"]["points"]:
        st.session_state["sel_box"] = selection["selection"]["points"][0]["hovertext"]

with col_info:
    selected = st.selectbox(
        "Buscar Jurisdicción:",
        options=jurisdicciones_lista,
        key="sel_box"
    )

    info_data = df_raw[df_raw[df_col_nombre] == selected]

    if not info_data.empty:
        info = info_data.iloc[0]
        
        st.subheader(f"Detalle: {selected}")
        st.metric("Grado de Peligrosidad", info["status"])
        
        delitos_vals = info[hover_cols].astype(float).sort_values(ascending=False)
        delitos_sin_otros = delitos_vals.drop(labels=["Otros"], errors="ignore")

        if delitos_vals.index[0] == "Otros" and len(delitos_sin_otros) > 0:
            delito_critico = delitos_sin_otros.idxmax()
        else:
            delito_critico = delitos_vals.idxmax()


        st.write("---")
        st.write("**Distribución de Delitos (Ponderados):**")
        st.bar_chart(delitos_vals)
        
        st.success(f"Delito crítico: {delito_critico}")

    else:
        st.info("Haz clic en un área del mapa o selecciona una opción de la lista.")

    st.write("---")
    st.subheader("Enviar Calificacion de la App")
    st.write("Tu feedback ayuda a mejorar los modelos.")

    # Formulario de calificación: input numérico 0-5 (evita reruns al mover controles)
    with st.form("rating_form"):
        stars_sel = st.number_input("Calificación (0-5)", min_value=0, max_value=5, value=3, step=1)
        comment = st.text_area("Comentario (opcional)", value="", max_chars=240)
        submit = st.form_submit_button("Enviar Calificación")

    if submit:
        try:
            ratings.add_rating(int(stars_sel), comment)
            st.success("Calificación guardada.")

            # Parámetros internos de disparo (no expuestos al usuario)
            low_star_threshold = 2
            trigger_count = 5

            if ratings.should_trigger_retrain(low_star_threshold=low_star_threshold, trigger_count=trigger_count):
                with st.spinner("Se han detectado varias calificaciones bajas. Reentrenando modelos... esto puede tardar varios minutos."):
                    ok = ratings.trigger_retrain()

                if ok:
                    st.success("Reentrenamiento completado. La app se recargará para mostrar los nuevos resultados.")
                    try:
                        load_data.clear()
                        load_geojson.clear()
                    except Exception:
                        pass
                    st.experimental_rerun()
                else:
                    st.error("Ocurrió un error durante el reentrenamiento. Revisa los logs del servidor.")
        except Exception as e:
            st.error(f"Error guardando la calificación: {e}")

st.divider()
st.caption(f"SafeZone v2.0 | Modelo GMM | Clústeres detectados: {n_clusters}")