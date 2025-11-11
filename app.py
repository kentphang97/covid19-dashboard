
# Core Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import plotly.express as px

# Preprocessing & Model Selection
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA

# Machine Learning Models
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor
)
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN

import scipy.cluster.hierarchy as sch

# Evaluation Metrics
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    r2_score, mean_absolute_error, mean_squared_error
)

# 2. LOAD THE COVID-19 DATASET
# ------------------------------------------------------------
df = pd.read_csv("./data.csv")
print("Dataset Loaded Successfully!")
print("Shape:", df.shape)
print(df.head())

# 3. DATA CLEANING AND EXPLORATORY DATA ANALYSIS (EDA)
# ------------------------------------------------------------
print("\n=== Missing Values ===")
print(df.isnull().sum())

# Replace NaN with 0 for simplicity
df.fillna(0, inplace=True)

# Check duplicates
print("\nDuplicate rows:", df.duplicated().sum())

# Info and Summary
df.info()

print(df.describe())

# --- Page Setup ---
st.set_page_config(page_title="COVID-19 Dashboard", layout="wide")

# Title and Introduction
# -------------------------------
st.title("🌍 COVID-19 Data Visualization Dashboard")
st.markdown("""
This interactive dashboard visualizes global COVID-19 data including **confirmed cases, recoveries, deaths, and regional trends**.
Use the filters and charts below to explore how COVID-19 has impacted different countries and regions.
""")
st.markdown("""Data was retrieved from https://www.kaggle.com/datasets/imdevskp/corona-virus-report?select=covid_19_clean_complete.csv which dated from 22nd January 2020 to 27nd July 2020.""")

# -------------------------------
# Global Overview
# -------------------------------
st.header("📊 Global Summary")

total_confirmed = int(df["Confirmed"].sum())
total_deaths = int(df["Deaths"].sum())
total_recovered = int(df["Recovered"].sum())
total_active = int(df["Active"].sum())

col1, col2, col3, col4 = st.columns(4)
col1.metric("Confirmed Cases", f"{total_confirmed:,}")
col2.metric("Deaths", f"{total_deaths:,}")
col3.metric("Recovered", f"{total_recovered:,}")
col4.metric("Active Cases", f"{total_active:,}")

st.markdown("---")

# -------------------------------
# Top 10 Countries
# -------------------------------
st.subheader("🏆 Top 10 Countries by Confirmed Cases")
top10 = df.sort_values(by="Confirmed", ascending=False).head(10)
fig_top10 = px.bar(
    top10,
    x="Country/Region",
    y="Confirmed",
    color="Confirmed",
    text="Confirmed",
    title="Top 10 Countries by Confirmed Cases"
)
fig_top10.update_traces(textposition='outside')
st.plotly_chart(fig_top10, use_container_width=True)

# -------------------------------
# Regional Analysis
# -------------------------------
st.subheader("🌐 Cases by WHO Region")
region_df = df.groupby("WHO Region")[["Confirmed", "Deaths", "Recovered"]].sum().reset_index()
fig_region = px.bar(
    region_df,
    x="WHO Region",
    y=["Confirmed", "Deaths", "Recovered"],
    barmode="group",
    title="COVID-19 Summary by WHO Region"
)
st.plotly_chart(fig_region, use_container_width=True)

# -------------------------------
# Interactive World Map
# -------------------------------
st.subheader("🗺️ Global Map View")
fig_map = px.scatter_geo(
    df,
    locations="Country/Region",
    locationmode="country names",
    size="Confirmed",
    color="Deaths",
    hover_name="Country/Region",
    title="COVID-19 Cases Around the World",
    projection="natural earth",
    color_continuous_scale="Reds"
)
st.plotly_chart(fig_map, use_container_width=True)

# -------------------------------
# Country Filter + Details (Default: Malaysia)
# -------------------------------
st.subheader("🔍 Country Details")

# Sort and define default
country_list = sorted(df["Country/Region"].unique())
default_country = "Malaysia"
default_index = country_list.index(default_country) if default_country in country_list else 0

# Create dropdown with default = Malaysia
country_choice = st.selectbox("Select a Country:", country_list, index=default_index)

# Retrieve data for selected country
country_data = df[df["Country/Region"] == country_choice].iloc[0]

# Display metrics
st.markdown(f"### COVID-19 Statistics for {country_choice}")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Confirmed", f"{country_data['Confirmed']:,}")
col2.metric("Deaths", f"{country_data['Deaths']:,}")
col3.metric("Recovered", f"{country_data['Recovered']:,}")
col4.metric("Active", f"{country_data['Active']:,}")

# -------------------------------
# Pie Chart for Selected Country (Default: Malaysia)
# -------------------------------
fig_pie = px.pie(
    names=["Active", "Recovered", "Deaths"],
    values=[country_data["Active"], country_data["Recovered"], country_data["Deaths"]],
    title=f"COVID-19 Distribution in {country_choice}"
)
st.plotly_chart(fig_pie, use_container_width=True)

# -------------------------------
# Trend Over Time (Optional)
# -------------------------------
st.subheader("1-Week Change In Confirmed Cases (Top10)")
st.markdown("""
If you have time-series data (daily updates), you can include a line chart to track cases over time.
This example uses 'Confirmed last week' and '1 week change' as a trend indicator.
""")

if "Confirmed last week" in df.columns and "1 week change" in df.columns:
    trend_df = df.sort_values("1 week change", ascending=False).head(10)
    fig_trend = px.bar(
        trend_df,
        x="Country/Region",
        y="1 week change",
        color="1 week change",
        title="1-Week Change in Confirmed Cases (Top 10)"
    )
    st.plotly_chart(fig_trend, use_container_width=True)
else:
    st.info("Time-series data not available in this CSV.")

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.caption("📊 Data Source: Public COVID-19 Dataset (country_wise_latest.csv)")
st.caption("Developed using Streamlit & Plotly")
'''
# Correlation Heatmap
plt.figure(figsize=(12,8))
sns.heatmap(data.select_dtypes(include=np.number).corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap of COVID-19 Variables")
plt.show()

# Distribution Plots
plt.figure(figsize=(12,6))
sns.histplot(data['Confirmed'], bins=30, kde=True)
plt.title("Distribution of Confirmed Cases")
plt.show()

sns.pairplot(data[['Confirmed', 'Deaths', 'Recovered', 'Active']])
plt.suptitle("Pairwise Relationships of Key COVID-19 Variables", y=1.02)
plt.show()

# EDA PIE CHART: Share of Confirmed Cases by Continent/Region
plt.figure(figsize=(6, 6))
data.groupby('WHO Region')['Confirmed'].sum().plot.pie(
    autopct='%1.1f%%',
    startangle=90,
    colors=plt.cm.Paired.colors,
    ylabel=''
)
plt.title('Share of Total Confirmed COVID-19 Cases by Region')
plt.show()

# EDA PIE CHART: Global COVID-19 Case Outcome Distribution
total_active = data['Active'].sum()
total_recovered = data['Recovered'].sum()
total_deaths = data['Deaths'].sum()

outcomes = [total_active, total_recovered, total_deaths]
labels = ['Active', 'Recovered', 'Deaths']
colors = ['#66b3ff', '#99ff99', '#ff6666']

plt.figure(figsize=(6, 6))
plt.pie(outcomes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
plt.title('Global COVID-19 Case Outcome Distribution')
plt.show()

# 4. FEATURE ENGINEERING
# ------------------------------------------------------------
# Create derived metrics
data['DeathRate'] = (data['Deaths'] / (data['Confirmed'] + 1)) * 100
data['RecoveryRate'] = (data['Recovered'] / (data['Confirmed'] + 1)) * 100
data['ActiveRatio'] = (data['Active'] / (data['Confirmed'] + 1)) * 100

# Create a binary label for classification (High vs Low Death Rate)
threshold = data['DeathRate'].mean()
data['HighDeathRate'] = (data['DeathRate'] > threshold).astype(int)

# Feature selection
features = ['Confirmed', 'Deaths', 'Recovered', 'Active', 'RecoveryRate', 'ActiveRatio']
target_class = 'HighDeathRate'
target_reg = 'Deaths'

# 5. DATA SPLITTING AND SCALING
# ------------------------------------------------------------
X = data[features]
y_class = data[target_class]
y_reg = data[target_reg]

X_train, X_test, y_train_class, y_test_class = train_test_split(X, y_class, test_size=0.2, random_state=42)
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X, y_reg, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_train_scaled_r = scaler.fit_transform(X_train_r)
X_test_scaled_r = scaler.transform(X_test_r)

"""# SUPERVISED - CLASSIFICATION"""

# 6. CLASSIFICATION MODELS
# Classification is used to categorize countries into groups such as high or low death rate based on COVID-19 data.
# It helps identify which countries are at greater risk by analyzing features like confirmed, recovered, and active cases.
# This supports decision-making for resource allocation and public health responses.
# ------------------------------------------------------------
models_class = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42)
    # "Decision Tree": DecisionTreeClassifier(random_state=42),
    # "KNN": KNeighborsClassifier(),
    # "Naive Bayes": GaussianNB(),
    # "SVM": SVC()

}

print("\n=== CLASSIFICATION RESULTS ===")
for name, model in models_class.items():
    model.fit(X_train_scaled, y_train_class)
    preds = model.predict(X_test_scaled)
    acc = accuracy_score(y_test_class, preds)
    print(f"\n{name} Accuracy: {acc:.4f}")
    print(classification_report(y_test_class, preds))

# Best model visualization (Gradient Boosting)
best_model = GradientBoostingClassifier(random_state=42)
best_model.fit(X_train_scaled, y_train_class)
y_pred_best = best_model.predict(X_test_scaled)

plt.figure(figsize=(6,5))
sns.heatmap(confusion_matrix(y_test_class, y_pred_best), annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix - Gradient Boosting Classifier")
plt.show()

"""# SUPERVISED - REGRESSION"""

# 7. REGRESSION MODELS
# Regression predicts continuous outcomes, such as the number of deaths, from other COVID-19 indicators.
# It reveals how variables like confirmed or recovered cases influence death counts.
# This enables forecasting and helps understand the strength of relationships between pandemic factors.
# ------------------------------------------------------------
models_reg = {
    "Linear Regression": LinearRegression(),
    "Decision Tree Regressor": DecisionTreeRegressor(random_state=42),
    "Gradient Boosting Regressor": GradientBoostingRegressor(random_state=42),
     "Random Forest Regressor": RandomForestRegressor(random_state=42)
    # "KNN Regressor": KNeighborsRegressor(),
    # "Support Vector Regressor": SVR(),
    # "Gradient Boosting Regressor": GradientBoostingRegressor(random_state=42)
}

print("\n=== REGRESSION RESULTS ===")
for name, model in models_reg.items():
    model.fit(X_train_scaled_r, y_train_r)
    preds = model.predict(X_test_scaled_r)
    print(f"\n{name}")
    print(f"R² Score: {r2_score(y_test_r, preds):.4f}")
    print(f"MAE: {mean_absolute_error(y_test_r, preds):.2f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test_r, preds)):.2f}")

# Compare True vs Predicted (Best model: Linear Regression)
rf_reg = LinearRegression()
rf_reg.fit(X_train_scaled_r, y_train_r)
y_pred_rf = rf_reg.predict(X_test_scaled_r)

plt.figure()
plt.scatter(y_test_r, y_pred_rf, alpha=0.7)
plt.plot([y_test_r.min(), y_test_r.max()], [y_test_r.min(), y_test_r.max()], 'r--')
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual vs Predicted - Linear Regression")
plt.show()

"""# UNSUPERVISED - CLUSTERING

"""

# Data Preparation
# ============================================================
# Select key numerical features for clustering
X_cluster = data[['Confirmed', 'Deaths', 'Recovered', 'Active', 'DeathRate', 'RecoveryRate']]

# Standardize data to ensure fair comparison across features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cluster)

# 8.1 K-MEANS CLUSTERING
# Purpose:
# K-Means groups countries into clusters based on COVID-19 metrics,
# helping to identify patterns like high-impact or low-recovery regions.
# ============================================================


# ------------------ Elbow Method ------------------
inertia = []
for k in range(2, 10):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8,5))
plt.plot(range(2, 10), inertia, marker='o')
plt.title("Elbow Method for Optimal Clusters")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia")
plt.show()

# ------------------ Apply Optimal K-Means ------------------
kmeans = KMeans(n_clusters=5, random_state=42)
data['KMeans_Cluster'] = kmeans.fit_predict(X_scaled)

# ------------------ Visualization ------------------
plt.figure(figsize=(10,6))
sns.scatterplot(data=data, x='Confirmed', y='Deaths', hue='KMeans_Cluster', palette='tab10')
plt.title("Country Clusters by COVID-19 Impact (K-Means)")
plt.xlabel("Confirmed Cases")
plt.ylabel("Deaths")
plt.legend(title="Cluster")
plt.show()

# 8.2 HIERARCHICAL CLUSTERING (AGGLOMERATIVE)
# Purpose:
# Hierarchical clustering builds a hierarchy of clusters using distance-based merging.
# The 'ward' linkage minimizes within-cluster variance for compact, well-separated groups.
# ============================================================


# ------------------ Apply Hierarchical Clustering ------------------
h_clus = AgglomerativeClustering(n_clusters=5, metric='euclidean', linkage='ward')
data['HCluster'] = h_clus.fit_predict(X_scaled)

# ------------------ Visualization ------------------
plt.figure(figsize=(10,6))
sns.scatterplot(data=data, x='Confirmed', y='Deaths', hue='HCluster', palette='tab10')
plt.title('Country Clusters by COVID-19 Impact (Hierarchical Clustering)')
plt.xlabel('Confirmed Cases')
plt.ylabel('Deaths')
plt.legend(title='Cluster')
plt.show()

# ------------------ Dendrogram ------------------
plt.figure(figsize=(14,6))
sch.dendrogram(
    sch.linkage(X_scaled, method='ward'),
    labels=data['Country/Region'].values,
    leaf_rotation=45,
    leaf_font_size=8
)
plt.title('Dendrogram for Hierarchical Clustering')
plt.xlabel('Countries')
plt.ylabel('Euclidean Distance')
plt.tight_layout()
plt.show()

# 8.3 DBSCAN CLUSTERING (Density-Based)
# Purpose:
# DBSCAN detects clusters of varying shapes and sizes based on data density.
# It identifies noise (outliers) and groups dense regions together,
# which helps reveal unusual COVID-19 patterns among countries.
# ============================================================


# ------------------ Apply DBSCAN on Scaled Data ------------------
dbscan = DBSCAN(eps=0.8, min_samples=4)  # tune eps if needed (0.5–1.5)
data['DBSCAN_Cluster'] = dbscan.fit_predict(X_scaled)

# ------------------ PCA Projection for Visualization ------------------
pca = PCA(n_components=2)
pca_data = pca.fit_transform(X_scaled)

plt.figure(figsize=(8,6))
plt.scatter(pca_data[:, 0], pca_data[:, 1], c=data['DBSCAN_Cluster'], cmap='plasma', s=70)
plt.title('DBSCAN Clustering of COVID-19 Data (PCA Projection)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()

# ------------------ Cluster Summary ------------------
print("Unique Cluster Labels:", set(data['DBSCAN_Cluster']))
print("Number of Clusters (excluding noise):", len(set(data['DBSCAN_Cluster'])) - (1 if -1 in data['DBSCAN_Cluster'] else 0))

# View Cluster Membership Summary
# ============================================================
print(data[['Country/Region', 'Confirmed', 'Deaths', 'KMeans_Cluster', 'HCluster', 'DBSCAN_Cluster']].head(10))

"""# VISUALIZATION & SUMMARY"""

# 9. INSIGHTS & SUMMARY
# ------------------------------------------------------------
top10 = data.sort_values(by="Confirmed", ascending=False).head(10)
sns.barplot(y="Country/Region", x="Confirmed", data=top10, palette="viridis")
plt.title("Top 10 Countries by Confirmed Cases")
plt.show()

sns.scatterplot(x="RecoveryRate", y="DeathRate", hue="KMeans_Cluster", data=data, palette="tab10")
plt.title("Recovery vs Death Rate by K-Means Cluster")
plt.show()

# 9. SUMMARY OF MODEL USED
# ------------------------------------------------------------
print("\n==============================")
print("MODEL PERFORMANCE USED")
print("==============================")
print("• Classification Models: Logistic, RF, GB")
print("• Regression Models: Linear, DT, GB")
print("• Clustering: K-Means (5 clusters), hierarchical, dbscan")
print("==============================")

'''





