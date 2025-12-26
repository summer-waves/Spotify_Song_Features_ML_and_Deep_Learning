# Spotify_Song_Features_ML_and_Deep_Learning

# 🎵 Spotify Track Popularity Prediction  
**Machine Learning & Deep Learning with Streamlit**

🔗 **Live Streamlit App:**  
https://spotifysongfeaturesmlanddeeplearning-jsappudcn4wwalt5rz93tmo.streamlit.app/

---

## 📑 Table of Contents
1. [Project Overview](#project-overview)
2. [Objectives](#objectives)
3. [Project Structure](#project-structure)
4. [Dataset & Features](#dataset--features)
5. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis(eda))
6. [Models Implemented](#models-implemented)
7. [Hyperparameter Tuning](#hyperparameter-tuning)
8. [Model Evaluation](#model-evaluation)
9. [Explainability with SHAP](#explainability-with-shap)
10. [Interactive Prediction](#interactive-prediction)
11. [Python Tools/Libraries/Modules](#python-tools/libraries/modules)
12. [Deployment](#deployment)
13. [Future Enhancements](#future-enhancements)

---

## 📌 Project Overview
This project predicts Spotify track popularity (0–100) using audio features provided by Spotify.
It combines exploratory data analysis (EDA), classical machine learning, ensemble methods, and deep learning, all wrapped inside a fully interactive Streamlit web application.

Users can:
- Explore the dataset visually
- Train and compare multiple models
- Perform hyperparameter tuning
- Generate predictions for new tracks
- Interpret model behavior using SHAP explainability

---

## 🎯 Objectives

- Understand which audio features influence track popularity
- Compare traditional ML models vs deep learning
- Build a production-style Streamlit app
- Practice model tuning, evaluation, and explainability
- Deploy a reproducible ML application to Streamlit Cloud

---

## 📂 Project Structure
- `spotify_song_features.py`
- `spotify_songs.csv`
- `requirements.txt`

---

## 📊 Dataset & Features

The dataset includes Spotify audio features such as:
- `danceability`
- `energy`
- `loudness`
- `speechiness`
- `acousticness`
- `instrumentalness`
- `liveness`
- `valence`
- `tempo`
- `duration_ms`

🎯 Target Variable:
`track_popularity` (0–100)

Data cleaning steps include:
- Removing duplicates
- Handling missing values
- Feature selection via Streamlit UI

---

## 🔍 Exploratory Data Analysis (EDA)

Inside the app, users can:

* Visualize popularity distributions
* Inspect feature correlations via heatmaps
* Create interactive scatter plots with OLS trendlines
* View summary statistics dynamically

📈 EDA is optional and toggleable to improve app performance.

---

## 🧠 Models Implemented
### Baseline & Linear Models

- Linear Regression
- Ridge Regression
- Lasso Regression

### Tree-Based & Ensemble Models

- Random Forest
- Extra Trees
- Gradient Boosting
- AdaBoost

### Distance & Kernel Methods

- K-Nearest Neighbors (KNN)
- Support Vector Regression (SVR)

### Deep Learning

- Fully connected Neural Network (TensorFlow/Keras)
- Standardization + Early Stopping

### Optional (Auto-Detected)

- XGBoost
- LightGBM
- CatBoost

Optional models are safely handled with `try/except` to avoid breaking deployment.

---

## ⚙️ Hyperparameter Tuning

This project implements `RandomizedSearchCV` with cross-validation for supported models.

Examples of tuned parameters:
- Tree depth, number of estimators
- Learning rates
- Regularization strength
- KNN neighbors
- SVR kernel parameters

Users can control:
- Number of CV folds
- Number of tuning iterations
- CPU parallelism (`n_jobs`)

---

## 📈 Model Evaluation

Each trained model is evaluated using:
- RMSE
- MAE
- MSE
- R² Score

Additional features:
- Actual vs Predicted plots
- Downloadable prediction CSVs
- Run comparison dashboard
- Model performance history

---

## 🧠 Explainability with SHAP

For tree-based models, the app provides:
- Global SHAP feature importance
- Fast sampling for performance
- Visual interpretation of feature influence

---

## 🔮 Interactive Prediction

Users can:
- Manually adjust audio features
- Generate real-time popularity predictions
- Clear or reuse previous predictions
- Test multiple trained models


---

## 🛠 Python Tools/Libraries/Modules

- Python
- Streamlit
- pandas / NumPy
- scikit-learn
- TensorFlow / Keras
- SHAP
- Matplotlib / Seaborn / Plotly
- statsmodels (for OLS trendlines)

---

## 🚀 Deployment

- Fully deployed on Streamlit Cloud
- Dependency-safe `requirements.txt`
- Environment conflicts resolved (TensorFlow + NumPy)
- Optimized for cloud performance

---

## 🧩 Key Takeaways

- Built an end-to-end ML pipeline
- Compared ML vs Deep Learning approaches
- Implemented real-world model tuning
- Designed a modular, scalable Streamlit app
- Solved real deployment and dependency issues
- Applied explainable AI concepts (SHAP)

---

## 📌 Future Enhancements

- Feature engineering (e.g., genre encoding)
- Classification version (Hit vs Non-Hit)
- Model persistence & reload
- User-uploaded CSV support
- Time-based popularity prediction






