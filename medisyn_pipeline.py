import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.svm import LinearSVC
from sklearn.decomposition import TruncatedSVD
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, FunctionTransformer, MinMaxScaler
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, silhouette_score

from sentence_transformers import SentenceTransformer
from xgboost import XGBClassifier

# -----------------------------
# Load Data
# -----------------------------
def load_data(path):
    COLS = [
        "Customer Identifier",
        "Medicine Name",
        "Rating: 10-star customer rating on overall satisfaction",
        "Effectiveness: 5 step effectiveness rating (categorical)",
        "Side Effects: 5 step side-effects rating (categorical)",
        "Name of condition/illness",
        "Customer Review of benefits (text)",
        "Customer Review of side effects (text)",
        "Overall Customer Review (text)"
        ]
    df = pd.read_csv(path, sep="\t", header=None, names=COLS)

    # Handle missing values
    for col in [
        "Overall Customer Review (text)",
        "Customer Review of benefits (text)",
        "Customer Review of side effects (text)",
        "Name of condition/illness"]:
        df[col] = df[col].fillna("")

    # Convert ratings to numeric safely
    df["Rating: 10-star customer rating on overall satisfaction"] = pd.to_numeric(
        df["Rating: 10-star customer rating on overall satisfaction"], errors="coerce"
        )
    
    # Drop rows where target labels are missing
    df = df.dropna(subset=[
        "Rating: 10-star customer rating on overall satisfaction",
        "Effectiveness: 5 step effectiveness rating (categorical)"
        ]
        )

    return df

# ------------------------
# Data Understanding
# ------------------------
def run_eda(dframe):
    # Check shape and missing values
    print("Shape: {}".format(dframe.shape))
    print("Null values: {}".format(dframe.isnull().sum()))

    # Effectiveness distribution
    plt.figure(figsize=(10,6))
    ax = sns.countplot(
        y=dframe["Effectiveness: 5 step effectiveness rating (categorical)"],
        order=dframe["Effectiveness: 5 step effectiveness rating (categorical)"].value_counts().index
    )

    # Title and labels with custom font sizes
    plt.title("Effectiveness Distribution", fontsize=16)
    plt.xlabel("Count", fontsize=14)
    plt.ylabel("Effectiveness Rating", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # Add text annotations to each bar
    for p in ax.patches:
        count = int(p.get_width())
        y_pos = p.get_y() + p.get_height() / 2
        ax.text(count - 10, y_pos, str(count), va='center', fontsize=12)

    plt.tight_layout()
    plt.show()

    # Rating distribution by Effectiveness
    plt.figure(figsize=(8,5))
    sns.boxplot(x="Effectiveness: 5 step effectiveness rating (categorical)", 
                y="Rating: 10-star customer rating on overall satisfaction", 
                data=dframe)
    plt.title("Rating vs Effectiveness", fontsize=16)
    plt.xlabel("Effectiveness", fontsize=14)
    plt.ylabel("Rating", fontsize=14)
    plt.xticks(fontsize=12, rotation=10)
    plt.yticks(fontsize=12)
    plt.show()

    # Review length distribution
    dframe["review_length"] = dframe["Customer Review of benefits (text)"].apply(lambda x: len(x.split()))
    plt.hist(dframe["review_length"].values, bins=50)
    plt.title("Distribution of Review Lengths (benefits)", fontsize=16)
    plt.xlabel("Number of Words", fontsize=14)
    plt.show()

    # Wordcloud for Highly Effective
    text = " ".join(dframe.loc[dframe["Effectiveness: 5 step effectiveness rating (categorical)"]=="Highly Effective", 
                               "Customer Review of benefits (text)"].dropna())
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title("Wordcloud - Highly Effective", fontsize=16)
    plt.show()

    # Wordcloud for Marginally Effective
    text = " ".join(dframe.loc[dframe["Effectiveness: 5 step effectiveness rating (categorical)"]=="Marginally Effective", 
                               "Customer Review of benefits (text)"].dropna())
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title("Wordcloud - Marginally Effective", fontsize=16)
    plt.show()

# ------------------------
# Sentiment Classifier
# ------------------------
def train_sentiment_classifier(train_df, test_df):
    X_train = train_df["Overall Customer Review (text)"]
    y_train = (train_df["Rating: 10-star customer rating on overall satisfaction"] >= 6).astype(int)
    
    X_test = test_df["Overall Customer Review (text)"]
    y_test = (test_df["Rating: 10-star customer rating on overall satisfaction"] >= 6).astype(int)
    
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=7000, stop_words="english", ngram_range=(1,3))),
        ("clf", LinearSVC(class_weight="balanced", random_state=42))
    ])
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    
    report = classification_report(y_test, preds, output_dict=True)
    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "precision_class_0": report["0"]["precision"],
        "recall_class_0": report["0"]["recall"],
        "f1_class_0": report["0"]["f1-score"],
        "precision_class_1": report["1"]["precision"],
        "recall_class_1": report["1"]["recall"],
        "f1_class_1": report["1"]["f1-score"]
        }
    
    return pipe, metrics

# ------------------------
# Effectiveness Predictor
# ------------------------
def train_effectiveness_model(train_df, test_df, model_type = "xgb"):
    target_col = "Effectiveness: 5 step effectiveness rating (categorical)"
    ORDER = ["Ineffective", "Marginally Effective", "Moderately Effective",
             "Considerably Effective", "Highly Effective"]

    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df["review_text"] = train_df["Customer Review of benefits (text)"]
    test_df["review_text"] = test_df["Customer Review of benefits (text)"]

    # Ordinal encode target
    le = LabelEncoder()
    train_df[target_col] = le.fit_transform(train_df[target_col])
    test_df[target_col] = le.transform(test_df[target_col])

    # SentenceTransformer
    sbert = SentenceTransformer("all-MiniLM-L6-v2")
    embedder = FunctionTransformer(lambda x: sbert.encode(x.tolist(), show_progress_bar=False))

    # Condition transformer (one-hot)
    condition_enc = OneHotEncoder(handle_unknown="ignore")

    # Preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ("review", embedder, "review_text"),
            ("condition", condition_enc, ["Name of condition/illness"]),
        ]
    )

    # Base models
    logreg = LogisticRegression(
        max_iter=500, class_weight="balanced", random_state=42
    )
    svm = LinearSVC(class_weight="balanced", random_state=42)
    xgb = XGBClassifier(
        objective="multi:softmax",
        num_class=len(ORDER),
        eval_metric="mlogloss",
        use_label_encoder=False,
        random_state=42
    )

    # classifier model selection
    if model_type == "logreg":
        clf_model = logreg
    elif model_type == "svm":
        clf_model = svm
    elif model_type == "xgb":
        clf_model = xgb
    elif model_type == "ensemble":
        clf_model = StackingClassifier(
            estimators=[("logreg", logreg), ("svm", svm), ("xgb", xgb)],
            final_estimator=LogisticRegression(
                max_iter=500, class_weight="balanced", random_state=42
            ),
            n_jobs=-1
        )
    else:
        raise ValueError("model_type must be 'logreg', 'svm', 'xgb', 'ordinal', or 'ensemble'")

    # Pipeline
    clf = Pipeline([
        ("features", preprocessor),
        ("clf", clf_model)
    ])

    # Train
    clf.fit(train_df, train_df[target_col])

    # Evaluate
    preds = clf.predict(test_df)
    acc = accuracy_score(test_df[target_col], preds)
    f1 = f1_score(test_df[target_col], preds, average="macro")
    report = classification_report(test_df[target_col], preds, 
                                   target_names=le.classes_, output_dict=True)

    metrics = {
        "accuracy": acc,
        "f1_macro": f1,
        "report": report
    }

    return clf, metrics

# ------------------------
# Customer Cohorts
# ------------------------
def cluster_customers(df, n_clusters=5):
    texts = df["Overall Customer Review (text)"].fillna("").astype(str)

    # Text -> TF-IDF -> SVD (to reduce dimensionality)
    text_pipe = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=3000, stop_words="english")),
        ("svd", TruncatedSVD(n_components=50, random_state=42))
    ])
    text_features = text_pipe.fit_transform(texts)

    # Add 10-star rating as numeric feature
    ratings = df["Rating: 10-star customer rating on overall satisfaction"]
    ratings = ratings.values.reshape(-1,1)
    features = np.hstack([text_features, ratings])

    # KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(features)
    df["Cluster"] = clusters

    # Cluster summaries
    cluster_summary = df.groupby("Cluster").agg({
        "Rating: 10-star customer rating on overall satisfaction": ["count", "mean"],
    }).round(2)
    silhouette = silhouette_score(features, clusters)
    
    return df, kmeans, cluster_summary, text_pipe, silhouette

# ------------------------
# Recommendation System
# ------------------------
def recommend_medicines(df, condition_query, top_k=5):
    df = df.dropna(subset=["Name of condition/illness", "Medicine Name"])
    conditions = df["Name of condition/illness"].astype(str)

    model = SentenceTransformer("all-MiniLM-L6-v2")
    cond_matrix = model.encode(conditions.tolist(), show_progress_bar=False)
    query_vec = model.encode([condition_query])

    sims = cosine_similarity(query_vec, cond_matrix).flatten()
    df["similarity"] = sims
    
    # Aggregate by medicine
    df["effectiveness_num"] = df["Effectiveness: 5 step effectiveness rating (categorical)"].map({
        "Ineffective": 1, 
        "Marginally Effective": 2, 
        "Moderately Effective": 3, 
        "Considerably Effective": 4, 
        "Highly Effective": 5
    })
    
    recs = (
        df.groupby("Medicine Name")
        .agg({
            "similarity": "mean",
            "Rating: 10-star customer rating on overall satisfaction": "mean",
            "effectiveness_num": "mean",
            "Medicine Name": "count"
            }
            )
            .rename(columns={"Medicine Name": "review_count"})
            .reset_index()
            )
    
    # Normalize scores ---
    scaler = MinMaxScaler()
    recs[["similarity", "Rating: 10-star customer rating on overall satisfaction", "effectiveness_num"]] = scaler.fit_transform(
        recs[["similarity", "Rating: 10-star customer rating on overall satisfaction", "effectiveness_num"]]
    )

    # Weighted score
    recs["score"] = (
        recs["similarity"] * 0.5 +
        recs["Rating: 10-star customer rating on overall satisfaction"] * 0.25 +
        recs["effectiveness_num"] * 0.25
    )

    # Explainability breakdown
    recs["explanation"] = recs.apply(lambda row: {
        "Condition similarity": round(row["similarity"], 3),
        "Avg satisfaction rating": round(row["Rating: 10-star customer rating on overall satisfaction"], 3),
        "Avg effectiveness": round(row["effectiveness_num"], 3),
        "Review count": int(row["review_count"]),
        "Final weighted score": round(row["score"], 3)
    }, axis=1)
    
    recs = recs.sort_values("score", ascending=False).head(top_k)

    return recs[["Medicine Name", "score", "explanation"]]

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    train_df = load_data(r"folder_for_interview\barclays\train.tsv")
    test_df = load_data(r"folder_for_interview\barclays\test.tsv")

    results = {}
    
    # 1. Sentiment
    clf_sent, metrics_sent = train_sentiment_classifier(train_df, test_df)
    results["sentiment"] = metrics_sent

    # 2. Effectiveness
    clf_eff, metrics_eff = train_effectiveness_model(train_df, test_df)
    results["effectiveness"] = metrics_eff

    # 3. Cohorts
    clustered_df, kmeans, cluster_summary, text_pipe, silhouette = cluster_customers(train_df, n_clusters=5)
    
    clustered_df.to_csv(r"folder_for_interview\barclays\cluster_summary.csv", index=False)

    # 4. Recommender
    recs = recommend_medicines(train_df, condition_query="flu", top_k=5)
    recs_dict = recs.to_dict(orient="records")
    with open(r"folder_for_interview\barclays\medicine_recommendation.json", "w") as f:
        json.dump(recs_dict, f, indent=4)
    
    # Save metrics
    with open(r"folder_for_interview\barclays\metrics_summary.json", "w") as f:
        json.dump(results, f, indent=4)
    
    print("Pipeline complete. Metrics, clusters, and recommendations saved.")
    print(json.dumps(results, indent=2))