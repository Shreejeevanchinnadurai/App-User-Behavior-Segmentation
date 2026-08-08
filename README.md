# 📊 App User Behavior Segmentation Using Unsupervised Machine Learning

A data science project that segments app users into behavioral groups using **K-Means clustering**, helping identify high-engagement, moderate, and at-risk users without any predefined labels.


---

## 📌 Problem Statement

Applications generate large amounts of user activity data, but understanding user behavior without predefined labels is difficult. This project analyzes app usage data using unsupervised machine learning to group users based on similar behavior and engagement patterns — helping identify high-engagement and at-risk users to improve retention and product decisions.

---

## 🧰 Tech Stack

| Category | Tools |
|---|---|
| Language | Python |
| Data Handling | Pandas, NumPy |
| Machine Learning | Scikit-learn (K-Means, PCA, StandardScaler) |
| Visualization | Matplotlib, Seaborn |
| Dashboard | Streamlit |

---

## 🔍 Approach

1. **Data Collection & Understanding** — loaded and explored a 50,000-row app user behavior dataset (25 columns: demographics, session metrics, engagement scores).
2. **Data Cleaning** — checked for duplicates (none found), handled missing values in `rating_given` (~10% missing, imputed with median), verified categorical consistency (gender, country, device_type, subscription_type, marketing_source).
3. **Feature Selection** — selected 14 behavioral/activity features for clustering (sessions, session duration, active minutes, page views, support tickets, etc.), excluding identifiers, demographics, and pre-derived scores (`engagement_score`, `churn_risk_score`) to avoid redundancy.
4. **Feature Scaling** — applied `StandardScaler` so all features contribute equally to distance-based clustering.
5. **Optimal Cluster Selection** — used the **Elbow Method** to determine k=4 as the optimal number of clusters.
6. **K-Means Clustering** — trained the model and assigned each user to one of 4 clusters.
7. **Model Evaluation** — validated cluster quality using a **correlation matrix**, **PCA 2D projection**, and **silhouette score**.
8. **Cluster Profiling** — analyzed average behavioral metrics per cluster and identified the top differentiating features.
9. **Business Insight Generation** — mapped each cluster to specific, actionable business recommendations.

---

## 📊 Results

The dataset was segmented into 4 clusters:

| Cluster | Size | Defining Trait |
|---|---|---|
| Short-Session Baseline | 17,338 users (34.7%) | Shortest sessions, fewest pages viewed, no support tickets |
| Higher-Support-Contact | 8,535 users (17.1%) | Notably higher support ticket volume |
| Long-Session | 7,107 users (14.2%) | ~3x longer average session duration than other clusters |
| High-Page-Depth Browser | 17,020 users (34.0%) | Highest pages viewed per session |

### ⚠️ Data Quality Finding

During evaluation, a correlation matrix across all 14 behavioral features (plus `engagement_score` and `churn_risk_score`) showed **near-zero correlation everywhere** (all pairwise correlations ≤ 0.02). PCA confirmed this — the first two principal components explained only **12.8%** of total variance, and the clusters showed heavy overlap in the 2D projection. The **silhouette score was 0.05**, confirming weak cluster separation.

**Interpretation:** this dataset does not contain a strong, holistic "engagement level" structure — most features are statistically independent of one another, including `engagement_score` and `churn_risk_score` relative to the raw behavioral metrics. K-Means still partitions the data (as any clustering algorithm will), but the resulting clusters reflect differences in a small subset of features (session duration, page depth, support contact) rather than distinct overall user personas. Clusters were labeled accordingly, based on their actual differentiating traits rather than a forced "engagement" narrative — this transparency is itself a key analytical finding of the project.

---

## 💼 Business Insights & Actions

| Cluster | Business Action |
|---|---|
| Short-Session Baseline | Largest segment — target with engagement nudges (notifications, personalized content) to increase activity |
| Higher-Support-Contact | Priority segment for UX friction investigation and proactive support outreach |
| Long-Session | Study what drives longer sessions; strong candidates for premium/power-user features |
| High-Page-Depth Browser | Improve in-app search/recommendations to reduce browsing friction |

---

## 📂 Project Structure

```
├── app.py                               # Streamlit dashboard (auto-loads dataset, displays full analysis)
├── app_user_behavior_dataset.csv        # Source dataset (50,000 users, 25 columns)
├── App_User_Behavior_Segmentation.ipynb # Full analysis notebook (EDA, cleaning, clustering, evaluation)
├── requirements.txt                     # Python dependencies
└── README.md
```

---

## ▶️ Run Locally

```bash
git clone https://github.com/Shreejeevanchinnadurai/App-User-Behavior-Segmentation.git
cd App-User-Behavior-Segmentation
pip install -r requirements.txt
python -m streamlit run app.py
```

The dashboard will open at `http://localhost:8501`.

---

## 📈 Evaluation Metrics Addressed

- **Data Quality & Preprocessing Accuracy** — duplicate checks, missing value imputation, categorical consistency validation
- **Feature Selection & Scaling** — behavioral features isolated from demographic/derived columns, StandardScaler applied
- **Cluster Quality & Separation** — evaluated via correlation matrix, PCA, and silhouette score (transparently reported)
- **Cluster Interpretability** — clusters labeled by their actual strongest differentiating traits
- **Business Relevance & Actionability** — each cluster mapped to a concrete recommendation
- **Scalability & Performance** — pipeline handles 50,000 rows efficiently, structured to scale further

---

## 👤 Author

**Shreejeevan Chinnadurai**
[LinkedIn](https://linkedin.com/in/shreejeevan-chinnadurai) · [GitHub](https://github.com/Shreejeevanchinnadurai) · [Portfolio](https://shreejeevanchinnadurai.github.io/portfolio)
