# Multimodel Spotify Track Popularity Prediction Using Audio Feature Engineering

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

The app allows users to:
- Explore and visualize audio feature data
- Train and tune multiple models
- Compare model performance across metrics
- Track and benchmark model runs
- Generate predictions for new tracks using trained models

---

## 🎯 Objectives

- Identify which audio features influence track popularity
- Compare traditional ML models with ensemble and deep learning approaches
- Build a production-style, interactive ML application
- Practice hyperparameter tuning and performance evaluation
- Deploy a reproducible ML application to Streamlit Cloud

---

## 📂 Project Structure
- **`spotify_song_features.py`**: The main streamlit application
- **`spotify_songs.csv`**: A dataset file containing all songs and their features, including measurements.
- **`requirements.txt`**: A text file of all Python libraries and modules utilized and deployed throughout the Spotify Streamlit project.

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

Optional models are only enabled when dependencies are available, ensuring stable deployment.

---

## ⚙️ Hyperparameter Tuning

Hyperparameter tuning is implemented using **`RandomizedSearchCV`** with cross-validation for supported models.

Examples of tuned parameters:
- Tree depth, number of estimators
- Learning rates
- Regularization strength
- KNN neighbor counts
- SVR kernel and margin parameters

Users can control:
- Number of **CV** (cross-validation) folds
- Number of tuning iterations
- CPU parallelism (`n_jobs`)

---

## 📈 Model Evaluation

Each trained model is evaluated using multiple metrics:
- RMSE
- MAE
- MSE
- R² Score
- MAPE
- Training time

Additional features:
- Actual vs. Predicted visualizations
- Downloadable prediction CSVs
- Persistent run history
- Side-by-side model comparison tables
- Composite Score that balances:
  * Predictive peformance (R^2, RMSE)
  * Computational efficiency (training time)
This enables identification of the **best overall model**, not just the most accurate one.

---

## 🔮 Interactive Prediction

Users can:
- Manually adjust audio feature values
- Generate real-time popularity predictions
- Clear or reuse previous predictions
- Test predictions across different trained models


---

## 🛠 Python Tools/Libraries/Modules

- Python
- Streamlit
- pandas / NumPy
- scikit-learn
- TensorFlow / Keras
- XGBoost/LightGBM/CatBoost
- Matplotlib / Seaborn / Plotly
- statsmodels (for OLS trendlines)

---

## 🚀 Deployment

- Fully deployed on **Streamlit Cloud**
- Dependency-safe **`requirements.txt`**
- Environment conflicts resolved (TensorFlow + NumPy)
- Optimized for cloud performance

---

## 🧩 Key Takeaways

- Built an end-to-end ML pipeline
- Compared classical ML, ensemble models, and deep learning
- Implemented real-world hyperparameter tuning
- Designed a modular, scalable Streamlit app
- Solved practical deployment and dependency challenges
- Applied rigorous model evaluation and benchmarking techniques

---

## 📌 Future Enhancements

- Feature engineering (e.g., genre encoding)
- Classification version (Hit vs. Non-Hit)
- Model persistence & reload
- User-uploaded CSV support
- Time-based popularity prediction






