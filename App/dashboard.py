from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.svm import OneClassSVM
from sklearn.cluster import DBSCAN

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "Dataset" / "ai4i2020.csv"
RESULTS_DIR = PROJECT_ROOT / "results"
COMPARISON_PATH = RESULTS_DIR / "model_comparison.csv"

RESULTS_DIR.mkdir(exist_ok=True)


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def add_industrial_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["temp_difference"] = (
        df["Process temperature [K]"] - df["Air temperature [K]"]
    )

    df["power_proxy"] = (
        df["Torque [Nm]"] * df["Rotational speed [rpm]"]
    )

    df["wear_torque"] = (
        df["Tool wear [min]"] * df["Torque [Nm]"]
    )

    df["speed_torque_ratio"] = (
        df["Rotational speed [rpm]"] /
        df["Torque [Nm]"].replace(0, np.nan)
    ).fillna(0)

    return df


def prepare_train_test_data(df: pd.DataFrame):
    target_column = "Machine failure"

    drop_columns = [
        "UDI",
        "Product ID",
        "Machine failure",
        "TWF",
        "HDF",
        "PWF",
        "OSF",
        "RNF",
    ]

    feature_columns = [col for col in df.columns if col not in drop_columns]

    X = df[feature_columns]
    y = df[target_column]

    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()
    numerical_features = X.select_dtypes(exclude=["object"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "preprocessor": preprocessor,
        "feature_columns": feature_columns,
    }


def train_supervised_models(prepared):
    X_train = prepared["X_train"]
    y_train = prepared["y_train"]
    preprocessor = prepared["preprocessor"]

    models = {
        "Logistic Regression": Pipeline([
            ("preprocessor", preprocessor),
            ("model", LogisticRegression(max_iter=1000, class_weight="balanced"))
        ]),

        "Random Forest": Pipeline([
            ("preprocessor", preprocessor),
            ("model", RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                class_weight="balanced"
            ))
        ]),
    }

    for model in models.values():
        model.fit(X_train, y_train)

    return models


def train_anomaly_models(prepared):
    X_train = prepared["X_train"]
    y_train = prepared["y_train"]
    preprocessor = prepared["preprocessor"]

    X_normal = X_train[y_train == 0]
    X_normal_scaled = preprocessor.fit_transform(X_normal)

    models = {
        "Isolation Forest": IsolationForest(
            contamination=0.04,
            random_state=42
        ),

        "One-Class SVM": OneClassSVM(
            kernel="rbf",
            nu=0.04,
            gamma="scale"
        ),

        "DBSCAN": DBSCAN(
            eps=2.5,
            min_samples=5
        )
    }

    models["Isolation Forest"].fit(X_normal_scaled)
    models["One-Class SVM"].fit(X_normal_scaled)
    models["DBSCAN"].fit(X_normal_scaled)

    return models

def evaluate_supervised_models(models, prepared):
    X_test = prepared["X_test"]
    y_test = prepared["y_test"]

    results = []

    for name, model in models.items():
        y_pred = model.predict(X_test)

        results.append({
            "Model": name,
            "Type": "Supervised",
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, zero_division=0),
            "Recall": recall_score(y_test, y_pred, zero_division=0),
            "F1-score": f1_score(y_test, y_pred, zero_division=0),
        })

    return results


def evaluate_anomaly_models(models, prepared):
    X_test = prepared["X_test"]
    y_test = prepared["y_test"]
    preprocessor = prepared["preprocessor"]

    X_test_scaled = preprocessor.transform(X_test)

    results = []

    for name, model in models.items():
        if name == "DBSCAN":
            y_pred = np.where(model.fit_predict(X_test_scaled) == -1, 1, 0)
        else:
            raw_pred = model.predict(X_test_scaled)
            y_pred = np.where(raw_pred == -1, 1, 0)

        results.append({
            "Model": name,
            "Type": "Unsupervised",
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, zero_division=0),
            "Recall": recall_score(y_test, y_pred, zero_division=0),
            "F1-score": f1_score(y_test, y_pred, zero_division=0),
        })

    return results


def save_model_comparison(results):
    comparison = pd.DataFrame(results)
    comparison.to_csv(COMPARISON_PATH, index=False)
    return comparison


@st.cache_resource
def train_dashboard_pipeline():
    raw_df = load_data(DATA_PATH)
    engineered_df = add_industrial_features(raw_df)

    prepared = prepare_train_test_data(engineered_df)

    supervised_models = train_supervised_models(prepared)
    anomaly_models = train_anomaly_models(prepared)

    results = []
    results.extend(evaluate_supervised_models(supervised_models, prepared))
    results.extend(evaluate_anomaly_models(anomaly_models, prepared))

    comparison = save_model_comparison(results)

    all_models = {}
    all_models.update(supervised_models)
    all_models.update(anomaly_models)

    return engineered_df, prepared, all_models, comparison


def predict_dashboard_row(model_name, model, row_features, prepared):
    preprocessor = prepared["preprocessor"]

    if model_name in ["Logistic Regression", "Random Forest"]:
        prediction = int(model.predict(row_features)[0])

        probability = None
        if hasattr(model, "predict_proba"):
            probability = float(model.predict_proba(row_features)[0, 1])

        return prediction, probability

    row_scaled = preprocessor.transform(row_features)

    if model_name == "DBSCAN":
        dbscan_result = model.fit_predict(row_scaled)
        prediction = 1 if dbscan_result[0] == -1 else 0
        return prediction, None

    raw_prediction = model.predict(row_scaled)[0]
    prediction = 1 if raw_prediction == -1 else 0
    anomaly_score = float(-model.decision_function(row_scaled)[0])

    return prediction, anomaly_score



def show_sensor_values(row):
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Air temperature", f"{row['Air temperature [K]']:.1f} K")
    col2.metric("Process temperature", f"{row['Process temperature [K]']:.1f} K")
    col3.metric("Rotational speed", f"{row['Rotational speed [rpm]']:.0f} rpm")
    col4.metric("Torque", f"{row['Torque [Nm]']:.1f} Nm")

    col5, col6, col7, col8 = st.columns(4)

    col5.metric("Tool wear", f"{row['Tool wear [min]']:.0f} min")
    col6.metric("Temperature difference", f"{row['temp_difference']:.1f} K")
    col7.metric("Power proxy", f"{row['power_proxy']:.0f}")
    col8.metric("Wear torque", f"{row['wear_torque']:.0f}")




def main():
    st.set_page_config(
        page_title="AI4I Predictive Maintenance",
        layout="wide"
    )

    st.title("Real-Time IoT Anomaly Detection for Industrial Predictive Maintenance")

    if not DATA_PATH.exists():
        st.error(f"Dataset not found: {DATA_PATH}")
        st.stop()

    df, prepared, models, comparison = train_dashboard_pipeline()

    st.sidebar.title("Dashboard Control")

    selected_model_name = st.sidebar.selectbox(
        "Choose model",
        list(models.keys()),
        index=1
    )

    selected_row = st.sidebar.slider(
        "Simulated IoT row",
        0,
        len(df) - 1,
        0
    )

    model = models[selected_model_name]
    row = df.iloc[selected_row]

    row_features = row[prepared["feature_columns"]].to_frame().T

    prediction, score = predict_dashboard_row(
        selected_model_name,
        model,
        row_features,
        prepared
    )

    st.subheader("Live Sensor Values")
    show_sensor_values(row)

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("ANOMALY ALERT: possible machine failure detected.")
    else:
        st.success("Normal operation predicted.")

    if score is not None:
        if selected_model_name in ["Isolation Forest", "One-Class SVM"]:
            st.caption(f"Anomaly score: {score:.4f}")
        else:
            st.progress(score)
            st.caption(f"Estimated failure probability: {score:.2%}")

    st.subheader("Current Row Features")
    st.dataframe(row_features, use_container_width=True)

    st.subheader("Model Comparison")
    st.dataframe(comparison, use_container_width=True)

    st.subheader("Dataset Preview")
    st.dataframe(df.head(20), use_container_width=True)


if __name__ == "__main__":
    main()
    
