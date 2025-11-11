
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
df['DeathRate'] = (df['Deaths'] / (df['Confirmed'] + 1)) * 100
df['RecoveryRate'] = (df['Recovered'] / (df['Confirmed'] + 1)) * 100
df['ActiveRatio'] = (df['Active'] / (df['Confirmed'] + 1)) * 100
threshold = df['DeathRate'].mean()
df['HighDeathRate'] = (df['DeathRate'] > threshold).astype(int)

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
st.title("COVID-19 Data Visualization Dashboard")
st.markdown("""
This interactive dashboard visualizes global COVID-19 data including **confirmed cases, recoveries, deaths, and regional trends**.
Use the filters and charts below to explore how COVID-19 has impacted different countries and regions.
""")
st.markdown("""Data was retrieved from https://www.kaggle.com/datasets/imdevskp/corona-virus-report?select=covid_19_clean_complete.csv which dated from 22nd January 2020 to 27nd July 2020.""")

# -------------------------------
# Global Overview
# -------------------------------
st.header("Global Summary")

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
st.subheader("Top 10 Countries by Confirmed Cases")
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
st.subheader("Cases by WHO Region")
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
st.subheader("Global Map View")
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
st.subheader("Country Details")

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
st.caption("Data Source: Public COVID-19 Dataset (country_wise_latest.csv)")
st.caption("Developed using Streamlit & Plotly")


# ===========================
# MAIN HEADER
# ===========================
st.title("🦠 COVID-19 Data Analytics & Machine Learning Dashboard")
st.markdown("""
This dashboard performs **Exploratory Data Analysis (EDA)**, **Classification**, **Regression**, and **Clustering**  
on the global COVID-19 dataset to uncover trends and insights.
""")

# ===========================
# TABS
# ===========================
tabs = st.tabs([
    "📊 EDA & Visualization",
    "🧠 Classification (Supervised)",
    "📈 Regression (Supervised)",
    "🧩 Clustering (Unsupervised)",
    "🧾 Summary Insights"
])

# ===========================
# TAB 1: EDA
# ===========================
with tabs[0]:
    st.header("📊 Exploratory Data Analysis")

    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(df.select_dtypes(include=np.number).corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    st.subheader("Distribution of Confirmed Cases")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(df["Confirmed"], bins=30, kde=True, ax=ax)
    st.pyplot(fig)

    st.subheader("Pairwise Relationships (Confirmed, Deaths, Recovered, Active)")
    fig = sns.pairplot(df[["Confirmed", "Deaths", "Recovered", "Active"]])
    st.pyplot(fig)

    st.subheader("Share of Total Confirmed Cases by WHO Region")
    fig, ax = plt.subplots(figsize=(6, 6))
    df.groupby("WHO Region")["Confirmed"].sum().plot.pie(
        autopct="%1.1f%%", startangle=90, colors=plt.cm.Paired.colors, ylabel='', ax=ax
    )
    st.pyplot(fig)

    st.subheader("Global COVID-19 Case Outcome Distribution")
    outcomes = [
        df["Active"].sum(),
        df["Recovered"].sum(),
        df["Deaths"].sum()
    ]
    labels = ["Active", "Recovered", "Deaths"]
    colors = ["#66b3ff", "#99ff99", "#ff6666"]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(outcomes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
    ax.set_title("Global COVID-19 Case Outcome Distribution")
    st.pyplot(fig)

# ===========================
# TAB 2: CLASSIFICATION
# ===========================
with tabs[1]:
    st.header("🧠 Supervised Learning: Classification")

    features = ['Confirmed', 'Deaths', 'Recovered', 'Active', 'RecoveryRate', 'ActiveRatio']
    target_class = 'HighDeathRate'

    X = df[features]
    y = df[target_class]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models_class = {
        "Logistic Regression": LogisticRegression(),
        "Random Forest": RandomForestClassifier(random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42)
    }

    results = []
    for name, model in models_class.items():
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)
        acc = accuracy_score(y_test, preds)
        results.append((name, acc))

    results_df = pd.DataFrame(results, columns=["Model", "Accuracy"])
    st.subheader("Model Performance Comparison")
    st.dataframe(results_df)

    best_model = GradientBoostingClassifier(random_state=42)
    best_model.fit(X_train_scaled, y_train)
    y_pred_best = best_model.predict(X_test_scaled)

    st.subheader("Confusion Matrix - Gradient Boosting Classifier")
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(confusion_matrix(y_test, y_pred_best), annot=True, fmt='d', cmap='Blues', ax=ax)
    st.pyplot(fig)

# ===========================
# TAB 3: REGRESSION
# ===========================
with tabs[2]:
    st.header("📈 Supervised Learning: Regression")

    target_reg = 'Deaths'
    X = df[features]
    y = df[target_reg]

    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled_r = scaler.fit_transform(X_train_r)
    X_test_scaled_r = scaler.transform(X_test_r)

    models_reg = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": RandomForestRegressor(random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(random_state=42)
    }

    results = []
    for name, model in models_reg.items():
        model.fit(X_train_scaled_r, y_train_r)
        preds = model.predict(X_test_scaled_r)
        results.append([name, r2_score(y_test_r, preds), mean_absolute_error(y_test_r, preds), np.sqrt(mean_squared_error(y_test_r, preds))])

    reg_results = pd.DataFrame(results, columns=["Model", "R² Score", "MAE", "RMSE"])
    st.dataframe(reg_results)

    st.subheader("Actual vs Predicted - Linear Regression")
    lr = LinearRegression()
    lr.fit(X_train_scaled_r, y_train_r)
    preds = lr.predict(X_test_scaled_r)
    fig, ax = plt.subplots()
    ax.scatter(y_test_r, preds, alpha=0.7)
    ax.plot([y_test_r.min(), y_test_r.max()], [y_test_r.min(), y_test_r.max()], 'r--')
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    st.pyplot(fig)

# ===========================
# TAB 4: CLUSTERING
# ===========================
with tabs[3]:
    st.header("🧩 Unsupervised Learning: Clustering")

    X_cluster = df[['Confirmed', 'Deaths', 'Recovered', 'Active', 'DeathRate', 'RecoveryRate']]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cluster)

    # --- KMeans ---
    inertia = []
    for k in range(2, 9):
        km = KMeans(n_clusters=k, random_state=42)
        km.fit(X_scaled)
        inertia.append(km.inertia_)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(2, 9), inertia, marker='o')
    ax.set_title("Elbow Method for Optimal K")
    st.pyplot(fig)

    kmeans = KMeans(n_clusters=5, random_state=42)
    df['KMeans_Cluster'] = kmeans.fit_predict(X_scaled)

    st.subheader("K-Means Clustering Result")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(data=df, x='Confirmed', y='Deaths', hue='KMeans_Cluster', palette='tab10', ax=ax)
    st.pyplot(fig)

    # --- Hierarchical Clustering ---
    h_clus = AgglomerativeClustering(n_clusters=5, metric='euclidean', linkage='ward')
    df['HCluster'] = h_clus.fit_predict(X_scaled)

    st.subheader("Hierarchical Clustering Result")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(data=df, x='Confirmed', y='Deaths', hue='HCluster', palette='tab10', ax=ax)
    st.pyplot(fig)

    st.subheader("Dendrogram")
    fig, ax = plt.subplots(figsize=(12, 6))
    sch.dendrogram(sch.linkage(X_scaled, method='ward'), ax=ax, leaf_rotation=45, leaf_font_size=8)
    st.pyplot(fig)

    # --- DBSCAN ---
    st.subheader("DBSCAN Clustering (Density-Based)")
    dbscan = DBSCAN(eps=0.8, min_samples=4)
    df['DBSCAN_Cluster'] = dbscan.fit_predict(X_scaled)

    pca = PCA(n_components=2)
    pca_df = pca.fit_transform(X_scaled)
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(pca_df[:, 0], pca_df[:, 1], c=df['DBSCAN_Cluster'], cmap='plasma', s=70)
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    st.pyplot(fig)

# ===========================
# TAB 5: SUMMARY
# ===========================
with tabs[4]:
    st.header("🧾 Insights & Summary")

    st.subheader("Top 10 Countries by Confirmed Cases")
    top10 = df.sort_values(by="Confirmed", ascending=False).head(10)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(y="Country/Region", x="Confirmed", data=top10, palette="viridis", ax=ax)
    st.pyplot(fig)

    st.subheader("Recovery vs Death Rate by K-Means Cluster")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(x="RecoveryRate", y="DeathRate", hue="KMeans_Cluster", data=df, palette="tab10", ax=ax)
    st.pyplot(fig)



