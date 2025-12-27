# ============================================================
# Spotify Track Popularity Prediction App
# ============================================================

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import pandas as pd
import streamlit as st

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    ExtraTreesRegressor,
    AdaBoostRegressor,
)
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

# Deep Learning (TensorFlow/Keras)
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping

warnings.filterwarnings("ignore")

# ----------------------------
# Optional heavy deps (safe imports)
# ----------------------------
XGBOOST_AVAILABLE = False
LGBM_AVAILABLE = False
CATBOOST_AVAILABLE = False

try:
    from xgboost import XGBRegressor  # type: ignore
    XGBOOST_AVAILABLE = True
except Exception:
    pass

try:
    from lightgbm import LGBMRegressor  # type: ignore
    LGBM_AVAILABLE = True
except Exception:
    pass

try:
    from catboost import CatBoostRegressor  # type: ignore
    CATBOOST_AVAILABLE = True
except Exception:
    pass


# ============================================================
# Streamlit Config
# ============================================================
st.set_page_config(
    page_title="Spotify Popularity Predictor (Pro)",
    page_icon="🎵",
    layout="wide",
)

st.title("🎵 Spotify Track Popularity Prediction")
st.write("Predict Spotify track popularity using ML + Deep Learning with EDA, tuning.")


# ============================================================
# Constants / Feature Selection
# ============================================================
DEFAULT_FEATURE_COLS = [
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    "duration_ms",
]
TARGET_COL = "track_popularity"


# ============================================================
# Sidebar Navigation + Global Controls
# ============================================================
st.sidebar.header("🧭 Navigation")

PAGE = st.sidebar.radio(
    "Go to",
    ["🏠 Home", "📊 EDA", "🧪 Train", "🔮 Predict", "📈 Compare", "⚙️ Settings"],
    index=0,
)

st.sidebar.header("⚙️ Global Settings")
seed = 42  # fixed seed for reproducibility

st.sidebar.header("🧩 Optional")
show_eda = st.sidebar.checkbox("Show Exploratory Data Analysis", value=True)
use_plotly = st.sidebar.toggle("Use Plotly charts (where possible)", value=True)


# ============================================================
# Data Load
# ============================================================
@st.cache_data
def load_data_from_path(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.drop_duplicates()
    df = df.dropna(subset=[TARGET_COL])
    return df

def get_data() -> pd.DataFrame:
    return load_data_from_path("spotify_songs.csv")

df = get_data()


# ============================================================
# Feature selection UI (lets you expand features later)
# ============================================================
st.sidebar.header("🧱 Features")
available_cols = list(df.columns)
default_features = [c for c in DEFAULT_FEATURE_COLS if c in available_cols]

FEATURE_COLS = st.sidebar.multiselect(
    "Select feature columns",
    options=[c for c in available_cols if c != TARGET_COL],
    default=default_features,
)

if TARGET_COL not in df.columns:
    st.error(f"Target column '{TARGET_COL}' not found in data.")
    st.stop()

if len(FEATURE_COLS) == 0:
    st.error("Please select at least 1 feature column.")
    st.stop()

X = df[FEATURE_COLS].copy()
y = df[TARGET_COL].copy()


# ============================================================
# Train/Test Split
# ============================================================
test_size = st.sidebar.slider("Test size", 0.1, 0.4, 0.2, 0.05)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=float(test_size), random_state=int(seed)
)


# ============================================================
# Session State helpers
# ============================================================
def ss_init():
    st.session_state.setdefault("trained", False)
    st.session_state.setdefault("bundle", {})          # stores model, scaler/pipeline, preds, metrics, etc.
    st.session_state.setdefault("history", [])         # list of runs for comparison

ss_init()


# ============================================================
# Utilities
# ============================================================
@dataclass
class RunResult:
    model_name: str
    mse: float
    rmse: float
    mae: float
    r2: float
    train_seconds: float
    extra: Dict[str, Any]

def compute_metrics(y_true, y_pred) -> Dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"mse": float(mse), "rmse": rmse, "mae": float(mae), "r2": float(r2)}

def download_button_from_df(label: str, df_out: pd.DataFrame, filename: str):
    csv = df_out.to_csv(index=False).encode("utf-8")
    st.download_button(label=label, data=csv, file_name=filename, mime="text/csv")

def is_tree_model(name: str) -> bool:
    return any(k in name for k in ["Random Forest", "Extra Trees", "Gradient Boosting", "XGBoost", "LightGBM", "CatBoost", "AdaBoost"])

def plot_actual_vs_pred(y_true, y_pred, title="Actual vs Predicted"):
    fig = px.scatter(
        x=y_true,
        y=y_pred,
        labels={"x": "Actual Popularity", "y": "Predicted Popularity"},
        title=title
    )
    fig.add_shape(
        type="line",
        x0=float(np.min(y_true)), y0=float(np.min(y_true)),
        x1=float(np.max(y_true)), y1=float(np.max(y_true)),
        line=dict(dash="dash")
    )
    st.plotly_chart(fig, use_container_width=True)


# ============================================================
# Model Registry (easy to expand to 1000 lines: add more models here)
# ============================================================
MODEL_CHOICES = [
    "Linear Regression (Baseline)",
    "Ridge Regression",
    "Lasso Regression",
    "Random Forest (Ensemble)",
    "Extra Trees (Ensemble)",
    "Gradient Boosting (Ensemble)",
    "AdaBoost (Ensemble)",
    "SVR (Kernel)",
    "KNN Regressor",
    "Neural Network (TensorFlow/Keras)",
]

if XGBOOST_AVAILABLE:
    MODEL_CHOICES.append("XGBoost (Optional)")
if LGBM_AVAILABLE:
    MODEL_CHOICES.append("LightGBM (Optional)")
if CATBOOST_AVAILABLE:
    MODEL_CHOICES.append("CatBoost (Optional)")


# ============================================================
# PAGE: Home
# ============================================================
if PAGE == "🏠 Home":
    st.subheader("🏠 Project Overview")
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Features selected", f"{len(FEATURE_COLS)}")
    c3.metric("Target", TARGET_COL)

    st.write("### Quick Actions")
    a1, a2, a3, a4 = st.columns(4)
    if a1.button("Go to EDA ➜"):
        st.session_state["__nav"] = "📊 EDA"
        st.rerun()
    if a2.button("Go to Train ➜"):
        st.session_state["__nav"] = "🧪 Train"
        st.rerun()
    if a3.button("Go to Predict ➜"):
        st.session_state["__nav"] = "🔮 Predict"
        st.rerun()
    if a4.button("Reset Session"):
        for k in ["trained", "bundle", "history"]:
            if k in st.session_state:
                del st.session_state[k]
        ss_init()
        st.success("Session reset.")
        st.rerun()

    st.write("### Data Preview")
    st.dataframe(df.head(20), use_container_width=True)


# ============================================================
# PAGE: EDA
# ============================================================
if PAGE == "📊 EDA":
    st.subheader("📊 Exploratory Data Analysis")

    if not show_eda:
        st.info("EDA is disabled from the sidebar checkbox.")
        st.stop()

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Popularity Distribution", "Correlation Heatmap", "Interactive Scatter", "Feature Stats"]
    )

    with tab1:
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.histplot(df[TARGET_COL], bins=30, kde=True, ax=ax)
        ax.set_title("Distribution of Track Popularity")
        ax.set_xlabel("Popularity")
        ax.set_ylabel("Count")
        st.pyplot(fig)

    with tab2:
        corr = df[FEATURE_COLS + [TARGET_COL]].corr(numeric_only=True)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(corr, cmap="coolwarm", annot=False, ax=ax)
        ax.set_title("Feature Correlation Matrix")
        st.pyplot(fig)

    with tab3:
        feature_x = st.selectbox("X-axis Feature", FEATURE_COLS, index=0)
        feature_y = st.selectbox("Y-axis Feature", FEATURE_COLS, index=min(1, len(FEATURE_COLS) - 1))

        # NOTE: trendline="ols" requires statsmodels
        fig = px.scatter(
            df,
            x=feature_x,
            y=feature_y,
            color=TARGET_COL,
            title="Audio Features with OLS Trendline",
            trendline="ols",
            opacity=0.6
        )
        st.plotly_chart(fig, use_container_width=True)

        st.caption("If you get `No module named 'statsmodels'`, install it in THIS environment: `pip install statsmodels` and restart Streamlit.")

    with tab4:
        st.write("Summary statistics for selected features:")
        st.dataframe(df[FEATURE_COLS + [TARGET_COL]].describe(), use_container_width=True)


# ============================================================
# PAGE: Train
# ============================================================
if PAGE == "🧪 Train":
    st.subheader("🧪 Model Training")

    st.sidebar.header("⚙️ Model Configuration")
    model_choice = st.sidebar.selectbox("Select Model", MODEL_CHOICES, index=0)

    tune = st.sidebar.toggle("Enable Hyperparameter Tuning", value=False)
    cv_folds = st.sidebar.slider("CV folds", 3, 10, 5, 1)
    n_iter = st.sidebar.slider("Random Search iterations", 5, 50, 10, 5)
    n_jobs = st.sidebar.selectbox("n_jobs", [-1, 1, 2, 4], index=1)

    train_btn = st.button("🚀 Train Model", type="primary")
    save_run_btn = st.button("➕ Save run to Compare")

    if train_btn:
        start = time.time()

        with st.spinner(f"Training: {model_choice}..."):

            best_params = None
            fitted_obj = None
            used_scaler = None
            preds = None

            # ------------------------
            # Linear Regression (Baseline)
            # ------------------------
            if model_choice == "Linear Regression (Baseline)":
                pipeline = Pipeline([
                    ("scaler", StandardScaler()),
                    ("model", LinearRegression())
                ])
                pipeline.fit(X_train, y_train)
                preds = pipeline.predict(X_test)
                fitted_obj = pipeline

            # Ridge
            elif model_choice == "Ridge Regression":
                model = Ridge(random_state=int(seed))
                pipeline = Pipeline([("scaler", StandardScaler()), ("model", model)])

                if tune:
                    param_dist = {"model__alpha": np.logspace(-3, 3, 50)}
                    search = RandomizedSearchCV(
                        pipeline, param_distributions=param_dist, n_iter=int(n_iter),
                        cv=int(cv_folds), scoring="r2", n_jobs=int(n_jobs), random_state=int(seed)
                    )
                    search.fit(X_train, y_train)
                    fitted_obj = search.best_estimator_
                    best_params = search.best_params_
                else:
                    pipeline.fit(X_train, y_train)
                    fitted_obj = pipeline

                preds = fitted_obj.predict(X_test)

            # Lasso
            elif model_choice == "Lasso Regression":
                model = Lasso(random_state=int(seed), max_iter=20000)
                pipeline = Pipeline([("scaler", StandardScaler()), ("model", model)])

                if tune:
                    param_dist = {"model__alpha": np.logspace(-4, 1, 60)}
                    search = RandomizedSearchCV(
                        pipeline, param_distributions=param_dist, n_iter=int(n_iter),
                        cv=int(cv_folds), scoring="r2", n_jobs=int(n_jobs), random_state=int(seed)
                    )
                    search.fit(X_train, y_train)
                    fitted_obj = search.best_estimator_
                    best_params = search.best_params_
                else:
                    pipeline.fit(X_train, y_train)
                    fitted_obj = pipeline

                preds = fitted_obj.predict(X_test)

            # ------------------------
            # Random Forest (Ensemble)
            # ------------------------
            elif model_choice == "Random Forest (Ensemble)":
                base = RandomForestRegressor(random_state=int(seed))
                if tune:
                    param_dist = {
                        "n_estimators": [100, 200],
                        "max_depth": [None, 10, 20],
                        "min_samples_split": [2, 5],
                        "min_samples_leaf": [1, 2, 4],
                        "max_features": ["sqrt", "log2", None],
                    }
                    search = RandomizedSearchCV(
                        base, param_distributions=param_dist, n_iter=int(n_iter),
                        cv=int(cv_folds), scoring="r2", n_jobs=int(n_jobs), random_state=int(seed)
                    )
                    search.fit(X_train, y_train)
                    fitted_obj = search.best_estimator_
                    best_params = search.best_params_
                else:
                    fitted_obj = base.fit(X_train, y_train)

                preds = fitted_obj.predict(X_test)

            # Extra Trees
            elif model_choice == "Extra Trees (Ensemble)":
                base = ExtraTreesRegressor(random_state=int(seed))
                if tune:
                    param_dist = {
                        "n_estimators": [100, 200],
                        "max_depth": [None, 10, 20],
                        "min_samples_split": [2, 5, 10],
                        "min_samples_leaf": [1, 2, 4],
                    }
                    search = RandomizedSearchCV(
                        base, param_distributions=param_dist, n_iter=int(n_iter),
                        cv=int(cv_folds), scoring="r2", n_jobs=int(n_jobs), random_state=int(seed)
                    )
                    search.fit(X_train, y_train)
                    fitted_obj = search.best_estimator_
                    best_params = search.best_params_
                else:
                    fitted_obj = base.fit(X_train, y_train)

                preds = fitted_obj.predict(X_test)

            # Gradient Boosting
            elif model_choice == "Gradient Boosting (Ensemble)":
                base = GradientBoostingRegressor(random_state=int(seed))
                if tune:
                    param_dist = {
                        "n_estimators": [100, 200],
                        "learning_rate": [0.05, 0.1],
                        "max_depth": [2, 3, 4],
                        "subsample": [0.7, 0.85, 1.0],
                    }
                    search = RandomizedSearchCV(
                        base, param_distributions=param_dist, n_iter=int(n_iter),
                        cv=int(cv_folds), scoring="r2", n_jobs=int(n_jobs), random_state=int(seed)
                    )
                    search.fit(X_train, y_train)
                    fitted_obj = search.best_estimator_
                    best_params = search.best_params_
                else:
                    fitted_obj = base.fit(X_train, y_train)

                preds = fitted_obj.predict(X_test)

            # AdaBoost
            elif model_choice == "AdaBoost (Ensemble)":
                base = AdaBoostRegressor(random_state=int(seed))
                if tune:
                    param_dist = {
                        "n_estimators": [50, 100],
                        "learning_rate": [0.05, 0.1],
                    }
                    search = RandomizedSearchCV(
                        base, param_distributions=param_dist, n_iter=int(n_iter),
                        cv=int(cv_folds), scoring="r2", n_jobs=int(n_jobs), random_state=int(seed)
                    )
                    search.fit(X_train, y_train)
                    fitted_obj = search.best_estimator_
                    best_params = search.best_params_
                else:
                    fitted_obj = base.fit(X_train, y_train)

                preds = fitted_obj.predict(X_test)

            # SVR
            elif model_choice == "SVR (Kernel)":
                pipeline = Pipeline([("scaler", StandardScaler()), ("model", SVR())])
                if tune:
                    param_dist = {
                        "model__C": np.logspace(-1, 3, 40),
                        "model__gamma": ["scale", "auto"],
                        "model__epsilon": np.linspace(0.05, 1.0, 20),
                        "model__kernel": ["rbf", "poly"],
                    }
                    search = RandomizedSearchCV(
                        pipeline, param_distributions=param_dist, n_iter=int(n_iter),
                        cv=int(cv_folds), scoring="r2", n_jobs=int(n_jobs), random_state=int(seed)
                    )
                    search.fit(X_train, y_train)
                    fitted_obj = search.best_estimator_
                    best_params = search.best_params_
                else:
                    pipeline.fit(X_train, y_train)
                    fitted_obj = pipeline

                preds = fitted_obj.predict(X_test)

            # KNN
            elif model_choice == "KNN Regressor":
                pipeline = Pipeline([("scaler", StandardScaler()), ("model", KNeighborsRegressor())])
                if tune:
                    param_dist = {
                        "model__n_neighbors": list(range(3, 51, 2)),
                        "model__weights": ["uniform", "distance"],
                        "model__p": [1, 2],
                    }
                    search = RandomizedSearchCV(
                        pipeline, param_distributions=param_dist, n_iter=int(n_iter),
                        cv=int(cv_folds), scoring="r2", n_jobs=int(n_jobs), random_state=int(seed)
                    )
                    search.fit(X_train, y_train)
                    fitted_obj = search.best_estimator_
                    best_params = search.best_params_
                else:
                    pipeline.fit(X_train, y_train)
                    fitted_obj = pipeline

                preds = fitted_obj.predict(X_test)

            # Neural Network (TensorFlow/Keras)
            elif model_choice == "Neural Network (TensorFlow/Keras)":
                used_scaler = StandardScaler()
                X_train_scaled = used_scaler.fit_transform(X_train)
                X_test_scaled = used_scaler.transform(X_test)

                nn = keras.Sequential([
                    layers.Dense(128, activation="relu", input_shape=(X_train_scaled.shape[1],)),
                    layers.Dropout(0.25),
                    layers.Dense(64, activation="relu"),
                    layers.Dropout(0.15),
                    layers.Dense(1),
                ])

                nn.compile(optimizer="adam", loss="mse", metrics=["mae"])

                early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

                nn.fit(
                    X_train_scaled,
                    y_train,
                    validation_split=0.2,
                    epochs=150,
                    batch_size=32,
                    callbacks=[early_stop],
                    verbose=0
                )

                preds = nn.predict(X_test_scaled).flatten()
                fitted_obj = nn

                st.subheader("🧠 Neural Network Architecture")
                nn.summary(print_fn=lambda x: st.text(x))

            # Optional XGBoost
            elif model_choice == "XGBoost (Optional)" and XGBOOST_AVAILABLE:
                base = XGBRegressor(
                    random_state=int(seed),
                    n_estimators=800,
                    learning_rate=0.05,
                    max_depth=5,
                    subsample=0.9,
                    colsample_bytree=0.9,
                )
                fitted_obj = base.fit(X_train, y_train)
                preds = fitted_obj.predict(X_test)

            # Optional LightGBM
            elif model_choice == "LightGBM (Optional)" and LGBM_AVAILABLE:
                base = LGBMRegressor(random_state=int(seed), n_estimators=1200, learning_rate=0.03)
                fitted_obj = base.fit(X_train, y_train)
                preds = fitted_obj.predict(X_test)

            # Optional CatBoost
            elif model_choice == "CatBoost (Optional)" and CATBOOST_AVAILABLE:
                base = CatBoostRegressor(
                    random_state=int(seed),
                    verbose=False,
                    iterations=1500,
                    learning_rate=0.03,
                    depth=6,
                )
                fitted_obj = base.fit(X_train, y_train)
                preds = fitted_obj.predict(X_test)

            else:
                st.error("Model choice not recognized or missing dependency.")
                st.stop()

        train_seconds = time.time() - start
        metrics = compute_metrics(y_test, preds)

        # Save bundle to session_state so Predict page won't “disappear”
        st.session_state["trained"] = True
        st.session_state["bundle"] = {
            "model_choice": model_choice,
            "model": fitted_obj,
            "scaler": used_scaler,         # None for non-NN pipeline models
            "preds": preds,
            "X_test": X_test,
            "y_test": y_test,
            "metrics": metrics,
            "best_params": best_params,
            "train_seconds": train_seconds,
            "features": FEATURE_COLS,
        }

        # UI
        st.success(f"Trained: {model_choice} in {train_seconds:.2f}s")
        if best_params is not None:
            st.info("Best Params")
            st.json(best_params)

        st.write("### 📊 Metrics")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("RMSE", f"{metrics['rmse']:.2f}")
        c2.metric("MAE", f"{metrics['mae']:.2f}")
        c3.metric("MSE", f"{metrics['mse']:.2f}")
        c4.metric("R²", f"{metrics['r2']:.3f}")

        st.write("### 📈 Actual vs Predicted")
        plot_actual_vs_pred(y_test, preds, title=f"Actual vs Predicted — {model_choice}")

        # Download predictions
        out_df = X_test.copy()
        out_df["actual"] = y_test.values
        out_df["predicted"] = preds
        download_button_from_df("⬇️ Download test predictions (CSV)", out_df, "spotify_predictions.csv")

        # Feature importance (tree models only)
        if is_tree_model(model_choice) and hasattr(fitted_obj, "feature_importances_"):
            st.write("### 🌲 Feature Importance")
            importances = fitted_obj.feature_importances_
            imp_df = pd.DataFrame({"Feature": FEATURE_COLS, "Importance": importances}).sort_values("Importance", ascending=False)
            fig = px.bar(imp_df, x="Importance", y="Feature", orientation="h", title=f"Feature Importance — {model_choice}")
            st.plotly_chart(fig, use_container_width=True)

    if save_run_btn and st.session_state.get("trained", False):
        b = st.session_state["bundle"]
        rr = RunResult(
            model_name=b["model_choice"],
            mse=b["metrics"]["mse"],
            rmse=b["metrics"]["rmse"],
            mae=b["metrics"]["mae"],
            r2=b["metrics"]["r2"],
            train_seconds=b["train_seconds"],
            extra={"best_params": b["best_params"]}
        )
        st.session_state["history"].append(rr.__dict__)
        st.success("Saved run to Compare page.")

    if st.session_state.get("trained", False):
        
        b = st.session_state["bundle"]
        
        st.markdown("---")
        st.subheader("📊 Model Performance")
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("RMSE", f"{b['metrics']['rmse']:.2f}")
        c2.metric("MAE", f"{b['metrics']['mae']:.2f}")
        c3.metric("MSE", f"{b['metrics']['mse']:.2f}")
        c4.metric("R²", f"{b['metrics']['r2']:.3f}")
        
        st.subheader("📈 Actual vs Predicted")
        plot_actual_vs_pred(
            b["y_test"],
            b["preds"],
            title=f"Actual vs Predicted — {b['model_choice']}"
        )
        
        if is_tree_model(b["model_choice"]) and hasattr(b["model"], "feature_importances_"):
            st.subheader("🌲 Feature Importance")
            
            imp_df = pd.DataFrame({
                "Feature": b["features"],
                "Importance": b["model"].feature_importances_
            }).sort_values("Importance", ascending=False)
            
            fig = px.bar(
                imp_df,
                x="Importance",
                y="Feature",
                orientation="h",
                title=f"Feature Importance — {b['model_choice']}"
            )
            st.plotly_chart(fig, use_container_width=True)


# ============================================================
# PAGE: Predict
# ============================================================
if PAGE == "🔮 Predict":
    st.subheader("🔮 Predict Track Popularity")

    if not st.session_state.get("trained", False):
        st.warning("Train a model first (go to 🧪 Train).")
        st.stop()

    b = st.session_state["bundle"]
    model_choice = b["model_choice"]
    model_obj = b["model"]
    scaler_obj = b["scaler"]
    features = b["features"]

    st.info(f"Using trained model: **{model_choice}**")

    left, right = st.columns([1, 1])

    with left:
        st.write("### Input Features")
        input_data = {}
        for col in features:
            # robust defaults
            col_min = float(df[col].min())
            col_max = float(df[col].max())
            col_mean = float(df[col].mean())
            input_data[col] = st.slider(col, col_min, col_max, col_mean)

        input_df = pd.DataFrame([input_data])
        st.dataframe(input_df, use_container_width=True)

    with right:
        st.write("### Actions")
        pred_btn = st.button("🔮 Predict Popularity", type="primary")
        clear_btn = st.button("🧹 Clear Prediction")

        if clear_btn:
            st.session_state.pop("last_pred", None)
            st.success("Cleared.")
            st.rerun()

        if pred_btn:
            # Pipeline models (sklearn) support .predict directly
            if isinstance(model_obj, Pipeline):
                pred = float(model_obj.predict(input_df)[0])
            else:
                # Neural network needs scaling
                if scaler_obj is not None:
                    x_scaled = scaler_obj.transform(input_df)
                    pred = float(model_obj.predict(x_scaled).flatten()[0])
                else:
                    pred = float(model_obj.predict(input_df)[0])

            st.session_state["last_pred"] = pred

        if "last_pred" in st.session_state:
            st.success(f"🎵 Predicted Popularity: **{st.session_state['last_pred']:.1f} / 100**")


# ============================================================
# PAGE: Compare
# ============================================================
if PAGE == "📈 Compare":
    st.subheader("📈 Compare Saved Runs")

    history = st.session_state.get("history", [])
    if not history:
        st.info("No saved runs yet. Train a model and click ➕ Save run to Compare.")
        st.stop()

    hist_df = pd.DataFrame(history)
    st.dataframe(hist_df, use_container_width=True)

    st.write("### Best by R²")
    best = hist_df.sort_values("r2", ascending=False).head(10)
    st.dataframe(best, use_container_width=True)

    download_button_from_df("⬇️ Download run history (CSV)", hist_df, "run_history.csv")


# ============================================================
# PAGE: Settings
# ============================================================
if PAGE == "⚙️ Settings":
    st.subheader("⚙️ App Settings / Diagnostics")

    st.write("### Environment Checks")
    st.code(
        "\n".join([
            f"XGBoost available: {XGBOOST_AVAILABLE}",
            f"LightGBM available: {LGBM_AVAILABLE}",
            f"CatBoost available: {CATBOOST_AVAILABLE}",
        ])
    )

    st.write("### Tips")
    st.markdown(
        """
- If Plotly trendline (`trendline="ols"`) errors: install **statsmodels** and restart Streamlit.
- If your **Predict** button “disappears”: use **session_state** (this app does).
- Too many dependencies can break Streamlit Cloud; keep optional imports behind try/except (this app does).
        """
    )