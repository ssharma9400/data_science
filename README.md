# Medisyn Pipeline

A machine learning pipeline for analyzing customer medicine reviews, predicting effectiveness, clustering customer cohorts, and recommending medicines.

---

## Features

- **Data Cleaning & EDA:** Loads tab-separated review data, handles missing values, and provides exploratory visualizations.
- **Sentiment Analysis:** Classifies customer review sentiment using TF-IDF and SVM.
- **Effectiveness Prediction:** Predicts medicine effectiveness (5-step categorical) using review text and condition info with multiple model options.
- **Customer Clustering:** Groups customers into cohorts using review text and satisfaction ratings.
- **Medicine Recommendation:** Recommends medicines for a given condition using semantic similarity and aggregated ratings.

---

## Pipeline Overview

1. **Load Data:** Cleans and prepares the dataset.
2. **Sentiment Classifier:** Predicts positive/negative sentiment from reviews.
3. **Effectiveness Predictor:** Predicts categorical effectiveness using review text and condition.
4. **Customer Cohorts:** Clusters customers based on review text and ratings.
5. **Recommendation System:** Suggests top medicines for a queried condition.
6. **Outputs:** Saves metrics, cluster summaries, and recommendations to disk.

---

## Outputs

- `metrics_summary.json`: Model metrics for sentiment and effectiveness.
- `cluster_summary.csv`: Customer cohort assignments.
- `medicine_recommendation.json`: Top medicine recommendations for a sample condition.

---

## Main Functions

- `load_data(path)`: Loads and cleans review data.
- `run_eda(dframe)`: Visualizes data distributions and review text.
- `train_sentiment_classifier(train_df, test_df)`: Trains and evaluates sentiment model.
- `train_effectiveness_model(train_df, test_df, model_type)`: Trains and evaluates effectiveness model (supports logistic regression, SVM, XGBoost, ensemble).
- `cluster_customers(df, n_clusters)`: Clusters customers using review text and ratings.
- `recommend_medicines(df, condition_query, top_k)`: Recommends medicines for a given condition.

---
