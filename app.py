import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

st.set_page_config(page_title="App User Behavior Segmentation", layout="wide")
sns.set_style("whitegrid")

DATA_PATH = "app_user_behavior_dataset.csv"   # CSV must sit in the same folder as app.py
K = 4                                          # fixed, based on Elbow Method analysis

FEATURES = [
    'sessions_per_week', 'avg_session_duration_min', 'daily_active_minutes',
    'feature_clicks_per_session', 'notifications_opened_per_week', 'in_app_search_count',
    'pages_viewed_per_session', 'crash_events_last_30_days', 'support_tickets_raised',
    'days_since_last_login', 'ads_clicked_last_30_days', 'content_downloads',
    'social_shares', 'account_age_days'
]

CLUSTER_NAMES = {
    0: 'Short-Session Baseline',
    1: 'Higher-Support-Contact',
    2: 'Long-Session',
    3: 'High-Page-Depth Browser'
}

st.title("📊 App User Behavior Segmentation")
st.caption("Unsupervised Machine Learning · K-Means Clustering · PCA")

# ---------------- Load & clean data ----------------
df = pd.read_csv(DATA_PATH)

n_before = len(df)
df = df.drop_duplicates()
if 'user_id' in df.columns:
    df = df.drop_duplicates(subset='user_id', keep='first')
n_after = len(df)

if 'rating_given' in df.columns and df['rating_given'].isnull().sum() > 0:
    df['rating_given'] = df['rating_given'].fillna(df['rating_given'].median())

# ---------------- Scale & cluster ----------------
X = df[FEATURES]
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=FEATURES, index=df.index)

kmeans = KMeans(n_clusters=K, init='k-means++', random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(X_scaled)
df['cluster_label'] = df['cluster'].map(CLUSTER_NAMES)

# ---------------- Overview metrics ----------------
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Users", f"{n_after:,}")
col2.metric("Duplicates Removed", n_before - n_after)
col3.metric("Features Used", len(FEATURES))
col4.metric("Clusters", K)

# ---------------- Cluster sizes ----------------
st.header("1️⃣ Cluster Sizes")
cluster_counts = df['cluster_label'].value_counts()
fig, ax = plt.subplots(figsize=(8, 4))
sns.barplot(x=cluster_counts.index, y=cluster_counts.values, hue=cluster_counts.index,
            palette='viridis', legend=False, ax=ax)
ax.set_xlabel("Cluster")
ax.set_ylabel("Number of Users")
plt.xticks(rotation=15)
st.pyplot(fig)

# ---------------- Data quality diagnostics ----------------
st.header("2️⃣ Data Quality Diagnostics")
st.markdown(
    "Before trusting cluster labels, we checked whether the dataset has genuine "
    "underlying structure — i.e. whether features actually correlate with each other."
)

diag_cols = FEATURES + [c for c in ['engagement_score', 'churn_risk_score'] if c in df.columns]
corr_matrix = df[diag_cols].corr()

c1, c2 = st.columns([2, 1])
with c1:
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, cmap='coolwarm', center=0, annot=False, ax=ax)
    ax.set_title("Feature Correlation Matrix")
    st.pyplot(fig)

with c2:
    max_off_diag = corr_matrix.where(~np.eye(len(corr_matrix), dtype=bool)).abs().max().max()
    sample_idx = df.sample(min(5000, len(df)), random_state=42).index
    sil_score = silhouette_score(X_scaled.loc[sample_idx], df.loc[sample_idx, 'cluster'])

    pca_diag = PCA(n_components=2)
    pca_diag.fit(X_scaled)
    variance_explained = pca_diag.explained_variance_ratio_.sum()

    st.metric("Max feature correlation", f"{max_off_diag:.2f}")
    st.metric("Silhouette Score", f"{sil_score:.3f}")
    st.metric("PCA variance (2D)", f"{variance_explained*100:.1f}%")

    if max_off_diag < 0.1 and sil_score < 0.15:
        st.warning(
            "⚠️ Low correlation and low silhouette score indicate weak natural "
            "clustering structure in this dataset. Clusters reflect the strongest "
            "differentiating features found, not fully distinct holistic personas."
        )
    else:
        st.success("Clusters show reasonable separation and feature correlation.")

# ---------------- PCA visualization ----------------
st.header("3️⃣ Cluster Visualization (PCA)")
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
df['pca1'] = X_pca[:, 0]
df['pca2'] = X_pca[:, 1]

fig, ax = plt.subplots(figsize=(9, 6))
sns.scatterplot(data=df, x='pca1', y='pca2', hue='cluster_label', palette='tab10', alpha=0.6, ax=ax)
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)")
ax.set_title("User Clusters — PCA 2D Projection")
st.pyplot(fig)

# ---------------- Cluster profiling ----------------
st.header("4️⃣ Cluster Profiling")
profile_cols = FEATURES.copy()
for extra in ['engagement_score', 'churn_risk_score', 'rating_given']:
    if extra in df.columns:
        profile_cols.append(extra)

cluster_profile = df.groupby('cluster_label')[profile_cols].mean().round(2)
st.dataframe(cluster_profile, use_container_width=True)

means_variance = cluster_profile.var().sort_values(ascending=False)
top_features = means_variance.head(3).index.tolist()

st.subheader("Top Differentiating Features")
for feat in top_features:
    fig, ax = plt.subplots(figsize=(8, 3.5))
    sns.barplot(data=df, x='cluster_label', y=feat, hue='cluster_label',
                palette='viridis', legend=False, ax=ax)
    ax.set_title(f"{feat.replace('_', ' ').title()} by Cluster")
    plt.xticks(rotation=15)
    st.pyplot(fig)

# ---------------- Business insights ----------------
st.header("5️⃣ Business Insights & Actions")
st.markdown("""
| Cluster | Insight | Business Action |
|---|---|---|
| **Short-Session Baseline** | Largest segment — shortest sessions, fewest pages, no support tickets | Target with engagement nudges (notifications, personalized content) |
| **Higher-Support-Contact** | Only cluster with meaningfully non-zero support tickets | Priority segment for UX friction investigation and proactive support outreach |
| **Long-Session** | ~3x longer average session duration than other clusters | Study what drives longer sessions; candidates for premium/power-user features |
| **High-Page-Depth Browser** | Highest pages viewed per session | Improve in-app search/recommendations to reduce browsing friction |
""")

# ---------------- Predict cluster for a new user ----------------
st.header("6️⃣ Predict Cluster for a New User")
st.markdown("Enter a user's behavioral stats below to see which segment they'd belong to.")

feature_display_names = {
    'sessions_per_week': 'Sessions per Week',
    'avg_session_duration_min': 'Avg Session Duration (min)',
    'daily_active_minutes': 'Daily Active Minutes',
    'feature_clicks_per_session': 'Feature Clicks per Session',
    'notifications_opened_per_week': 'Notifications Opened per Week',
    'in_app_search_count': 'In-App Search Count',
    'pages_viewed_per_session': 'Pages Viewed per Session',
    'crash_events_last_30_days': 'Crash Events (Last 30 Days)',
    'support_tickets_raised': 'Support Tickets Raised',
    'days_since_last_login': 'Days Since Last Login',
    'ads_clicked_last_30_days': 'Ads Clicked (Last 30 Days)',
    'content_downloads': 'Content Downloads',
    'social_shares': 'Social Shares',
    'account_age_days': 'Account Age (Days)'
}

with st.form("predict_form"):
    input_cols = st.columns(2)
    user_input = {}
    for i, feat in enumerate(FEATURES):
        col = input_cols[i % 2]
        default_val = float(df[feat].median())
        min_val = float(df[feat].min())
        max_val = float(df[feat].max())
        user_input[feat] = col.number_input(
            feature_display_names.get(feat, feat),
            min_value=min_val, max_value=max_val, value=default_val
        )
    submitted = st.form_submit_button("Predict Cluster")

if submitted:
    input_df = pd.DataFrame([user_input])[FEATURES]
    input_scaled = scaler.transform(input_df)
    predicted_cluster = kmeans.predict(input_scaled)[0]
    predicted_label = CLUSTER_NAMES.get(predicted_cluster, f"Cluster {predicted_cluster}")

    st.success(f"🎯 Predicted Segment: **{predicted_label}**")

    st.markdown("**How this user compares to the cluster average:**")
    compare_df = pd.DataFrame({
        'This User': user_input,
        f'{predicted_label} Avg': cluster_profile.loc[predicted_label, FEATURES].to_dict()
    })
    st.dataframe(compare_df.round(2), use_container_width=True)

# ---------------- Export ----------------
st.header("7️⃣ Export Cluster Assignments")
export_cols = ['user_id', 'cluster', 'cluster_label'] + [c for c in
    ['avg_session_duration_min', 'pages_viewed_per_session', 'support_tickets_raised',
     'engagement_score', 'churn_risk_score'] if c in df.columns]
export_df = df[export_cols]
st.download_button(
    "Download user_cluster_assignments.csv",
    data=export_df.to_csv(index=False).encode('utf-8'),
    file_name="user_cluster_assignments.csv",
    mime="text/csv"
)
st.dataframe(export_df.head(50), use_container_width=True)