"""Entrenamiento del modelo para el CSV de CardioGoodFitness."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DEFAULT_DATA_PATH = Path("CardioGoodFitness.csv")
DEFAULT_MODEL_PATH = Path("cardio_model.joblib")
TARGET_COLUMN = "Product"
CATEGORICAL_COLUMNS = ["Gender", "MaritalStatus"]


def load_dataset(csv_path: Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Carga el CSV y valida que contenga las columnas esperadas."""
    if not csv_path.exists():
        raise FileNotFoundError(f"No existe el archivo del dataset: {csv_path}")

    df = pd.read_csv(csv_path)
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
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Faltan columnas en el CSV: {sorted(missing_columns)}")
    return df


def build_model_from_dataframe(df: pd.DataFrame) -> dict:
    """Entrena un modelo para predecir el producto usando variables mixtas."""
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    numeric_columns = [column for column in X.columns if column not in CATEGORICAL_COLUMNS]
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), numeric_columns),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLUMNS),
        ]
    )

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=1,
        random_state=42,
        class_weight="balanced",
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    artifact = {
        "model": pipeline,
        "feature_columns": list(X.columns),
        "class_labels": sorted(y.unique().tolist()),
        "metrics": {
            "accuracy": float(accuracy),
            "samples": int(len(df)),
        },
    }
    return artifact


def train_model(csv_path: Path = DEFAULT_DATA_PATH, output_path: Path = DEFAULT_MODEL_PATH) -> dict:
    """Genera el artefacto del modelo y lo guarda en disco."""
    df = load_dataset(csv_path)
    artifact = build_model_from_dataframe(df)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, output_path)
    print(f"Modelo entrenado con accuracy: {artifact['metrics']['accuracy']:.3f}")
    print(f"Archivo guardado en: {output_path.resolve()}")
    return artifact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entrena un modelo para CardioGoodFitness.csv")
    parser.add_argument("--csv-path", type=Path, default=DEFAULT_DATA_PATH, help="Ruta del CSV de entrada")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH, help="Ruta del archivo joblib de salida")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_model(csv_path=args.csv_path, output_path=args.model_path)


if __name__ == "__main__":
    main()
