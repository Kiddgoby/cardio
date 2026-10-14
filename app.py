"""Aplicación Streamlit para predecir el producto de CardioGoodFitness."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

DATA_PATH = Path("CardioGoodFitness.csv")
MODEL_PATH = Path("cardio_model.joblib")
FEATURE_COLUMNS = [
    "Age",
    "Gender",
    "Education",
    "MaritalStatus",
    "Usage",
    "Fitness",
    "Income",
    "Miles",
]


@st.cache_data
def load_dataset(data_path: Path = DATA_PATH) -> pd.DataFrame:
    """Carga y valida el dataset del proyecto."""
    if not data_path.exists():
        raise FileNotFoundError(f"No existe el archivo {data_path}")

    df = pd.read_csv(data_path)
    required_columns = {
        "Product",
        "Age",
        "Gender",
        "Education",
        "MaritalStatus",
        "Usage",
        "Fitness",
        "Income",
        "Miles",
    }
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas en el CSV: {sorted(missing)}")
    return df


@st.cache_resource
def load_or_train_model(data_path: Path = DATA_PATH, model_path: Path = MODEL_PATH):
    """Carga el artefacto existente o lo crea a partir del CSV."""
    if model_path.exists():
        artifact = joblib.load(model_path)
        if isinstance(artifact, dict) and "model" in artifact:
            return artifact

    from train import build_model_from_dataframe

    df = load_dataset(data_path)
    artifact = build_model_from_dataframe(df)
    joblib.dump(artifact, model_path)
    return artifact


st.set_page_config(page_title="CardioGoodFitness Predictor", page_icon="🏋️", layout="wide")

st.title("🏋️ Predicción de producto CardioGoodFitness")
st.caption("Este modelo usa el CSV de CardioGoodFitness para estimar qué producto probablemente compraría una persona.")

try:
    df = load_dataset(DATA_PATH)
    artifact = load_or_train_model(DATA_PATH, MODEL_PATH)
    model = artifact["model"]
    metrics = artifact["metrics"]
except Exception as exc:  # pragma: no cover - interfaz de error amigable
    st.error(f"No se pudo cargar el dataset o entrenar el modelo: {exc}")
    st.stop()

product_counts = df["Product"].value_counts().rename_axis("Producto").reset_index(name="Cantidad")

col1, col2, col3 = st.columns(3)
col1.metric("Registros", len(df))
col2.metric("Productos", df["Product"].nunique())
col3.metric("Accuracy del modelo", f"{metrics.get('accuracy', 0):.3f}")

st.subheader("📊 Distribución del dataset")
st.bar_chart(product_counts.set_index("Producto"), use_container_width=True)

st.sidebar.header("⚙️ Datos del cliente")
user_input = {
    "Age": st.sidebar.number_input("Edad", min_value=18, max_value=80, value=35),
    "Gender": st.sidebar.selectbox("Género", ["Male", "Female"]),
    "Education": st.sidebar.slider("Años de educación", min_value=8, max_value=20, value=14),
    "MaritalStatus": st.sidebar.selectbox("Estado civil", ["Single", "Partnered"]),
    "Usage": st.sidebar.slider("Uso semanal", min_value=1, max_value=7, value=3),
    "Fitness": st.sidebar.slider("Nivel de fitness", min_value=1, max_value=5, value=3),
    "Income": st.sidebar.number_input("Ingresos", min_value=0, max_value=200000, value=50000, step=1000),
    "Miles": st.sidebar.number_input("Millas por semana", min_value=0, max_value=500, value=80, step=10),
}

input_frame = pd.DataFrame([user_input], columns=FEATURE_COLUMNS)
probabilities = model.predict_proba(input_frame)[0]
product_labels = model.classes_
probability_frame = pd.DataFrame(
    {
        "Producto": [str(label) for label in product_labels],
        "Probabilidad": [float(value) for value in probabilities],
    }
)

predicted_product = model.predict(input_frame)[0]
confidence = max(probabilities)

st.subheader("🎯 Predicción del producto")
st.success(f"Producto predicho: {predicted_product}")
st.write(f"Confianza estimada: **{confidence:.2%}**")

st.bar_chart(probability_frame.set_index("Producto"), use_container_width=True)

st.subheader("🧾 Datos introducidos")
st.dataframe(input_frame, use_container_width=True)

with st.expander("📘 Vista previa del CSV"):
    st.dataframe(df.head(15), use_container_width=True)
