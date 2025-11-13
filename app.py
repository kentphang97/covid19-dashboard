
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#......ML Models
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    r2_score, mean_absolute_error, mean_squared_error
)

#......STREAMLIT CONFIGURATION

st.set_page_config(page_title="COVID-19 Data Analysis", layout="wide")
st.title("COVID-19 Data Analysis Dashboard")
st.markdown("Interactive Data Visualization and Machine Learning Insights")
st.markdown("""Data was retrieved from https://www.kaggle.com/datasets/imdevskp/corona-virus-report?select=covid_19_clean_complete.csv which dated from 22nd January 2020 to 27nd July 2020.""")

#>>>>>>>>>>>>>>>>>>>>1.LOAD FIXED DATA

data = pd.read_csv("./data.csv")
#replace data source file location

#data cleaning -Replace NaN with 0 for simplicity
data.fillna(0, inplace=True)
st.success("Dataset Loaded Successfully!")
st.write("**Data Shape:**", data.shape)

st.header("Global Summary")

total_confirmed = int(data["Confirmed"].sum())
total_deaths = int(data["Deaths"].sum())
total_recovered = int(data["Recovered"].sum())
total_active = int(data["Active"].sum())

col1, col2, col3, col4 = st.columns(4)
col1.metric("Confirmed Cases", f"{total_confirmed:,}")
col2.metric("Deaths", f"{total_deaths:,}")
col3.metric("Recovered", f"{total_recovered:,}")
col4.metric("Active Cases", f"{total_active:,}")

st.markdown("---")
#.....SIDEBAR CONTROLS
#bootstrap, emojipedia

st.sidebar.header("Controls")

regions = ["All"] + sorted(data["WHO Region"].dropna().unique().tolist())
selected_region = st.sidebar.selectbox("Select WHO Region", regions)

if selected_region != "All":
    filtered_data = data[data["WHO Region"] == selected_region]
else:
    filtered_data = data.copy()

countries = ["All"] + sorted(filtered_data["Country/Region"].dropna().unique().tolist())
selected_country = st.sidebar.selectbox("Select Country", countries)

if selected_country != "All":
    filtered_data = filtered_data[filtered_data["Country/Region"] == selected_country]

#......Model Type
st.sidebar.markdown("---")
model_type = st.sidebar.radio("Select Task", ["Classification"])

#.......Model Selection
st.sidebar.markdown("### Model Selection")
class_model_name = st.sidebar.selectbox(
    "Classification Model", ["Logistic Regression", "Naive Bayes (NB)", "Support Vector Machine (SVM)"]
)

#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>2.FEATURE ENGINEERING

data['DeathRate'] = (data['Deaths'] / (data['Confirmed'] + 1)) * 100
data['RecoveryRate'] = (data['Recovered'] / (data['Confirmed'] + 1)) * 100
data['ActiveRatio'] = (data['Active'] / (data['Confirmed'] + 1)) * 100
threshold = data['DeathRate'].mean()
data['HighDeathRate'] = (data['DeathRate'] > threshold).astype(int)

#>>>>>>>>>>>>>>>>>>>>>>>>>>>>3.EDA VISUALIZATIONS

#....ALL countries based on Confirmed cases (Streamlit table replacement)
st.subheader("Interactive Table")
topALL_confirmed = data.sort_values('Confirmed', ascending=False)
topALL_confirmed_table = topALL_confirmed[['Country/Region', 'Confirmed', 'Deaths', 'Recovered', 'Active',
                                         'New cases', 'New deaths', 'New recovered', 'Deaths / 100 Cases',
                                         'Recovered / 100 Cases', 'Deaths / 100 Recovered',
                                         'Confirmed last week', '1 week change', '1 week % increase', 'WHO Region']]
st.dataframe(topALL_confirmed_table, width='stretch')

#st.subheader("Summary Statistics")
#st.write(data.describe())
st.subheader("Summary Statistics")
data.replace([np.inf, -np.inf], np.nan, inplace=True)
data.fillna(0, inplace=True)
st.write(data.describe())

#.....MAP VISUALIZATION
st.subheader("Global COVID-19 Cases Map")

# Ensure location columns exist (some datasets use different names)
if 'Country/Region' in data.columns:
    map_data = data.copy()

    # Replace missing coordinates if available or use country centroids via Plotly’s built-in geo mapping
    fig = px.scatter_geo(
        map_data,
        locations="Country/Region",           
        locationmode="country names",
        size="Confirmed",                     
        color="Deaths",                       
        hover_name="Country/Region",
        hover_data={
            "Confirmed": True,
            "Deaths": True,
            "Recovered": True,
            "Active": True,
            "RecoveryRate": ':.2f',
            "DeathRate": ':.2f'
        },
        color_continuous_scale="Reds",
        size_max=40,
        projection="natural earth",
        title="Global COVID-19 Spread (Bubble Size = Confirmed Cases, Color = Deaths)",
        width=1000,
        height=600
    )

    fig.update_layout(
        geo=dict(
            showframe=False,
            showcoastlines=True,
            projection_type='natural earth'
        )
    )

    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("No valid location data found for map visualization.")

#......Distribution
fig, ax = plt.subplots(figsize=(12, 6))
sns.histplot(data['Confirmed'], bins=30, kde=True, ax=ax)
ax.set_title("Distribution of Confirmed Cases")
st.pyplot(fig)

#.....Pairplot (sampled for speed)
st.subheader("Pairwise Relationships of Key COVID-19 Variables (sampled)")
pairplot_fig = sns.pairplot(
    data.sample(min(200, len(data)))[['Confirmed', 'Deaths', 'Recovered', 'Active']]
)
st.pyplot(pairplot_fig.fig)

#.....Correlation Heatmap
st.subheader("Correlation Heatmap")
fig, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(filtered_data.select_dtypes(include=np.number).corr(), annot=True, cmap="coolwarm", ax=ax)
st.pyplot(fig)

#......Sunburst
st.subheader("Confirmed Cases by WHO Region and Country")
fig = px.sunburst(
    filtered_data,
    path=['WHO Region', 'Country/Region'],
    values='Confirmed',
    color='WHO Region',
    color_discrete_sequence=px.colors.qualitative.Set3,
    width=500,     # =figsize width, height
    height=500     
)
st.plotly_chart(fig, width='stretch')

#......Global Pie Chart
st.subheader("Global COVID-19 Case Outcome Distribution")
total_active = data['Active'].sum()
total_recovered = data['Recovered'].sum()
total_deaths = data['Deaths'].sum()
labels = ['Active', 'Recovered', 'Deaths']
values = [total_active, total_recovered, total_deaths]
colors = ['#66b3ff', '#99ff99', '#ff6666']

fig = px.pie(
    names=labels,
    values=values,
    color=labels,
    color_discrete_sequence=colors,
    #title="Global COVID-19 Case Outcome Distribution",
    hole=0 #for a normal pie chart; set >0 for donut chart
)

#hover info
fig.update_traces(
    hoverinfo='label+percent+value',
    textinfo='percent+label',
    textfont_size=14,
    pull=[0.05, 0.05, 0.05]  #explode effect
)
st.plotly_chart(fig, width='stretch')

#.......Recovered cases by Country
st.subheader("Recovered COVID-19 Cases by Country")
data_sorted = data.sort_values('Recovered', ascending=True)
fig = px.bar(
    data_sorted,
    x='Recovered',
    y='Country/Region',
    orientation='h',
    color='Recovered',
    color_continuous_scale='rainbow'
    #title='Number of Recovered COVID-19 Cases by Country'
)
fig.update_layout(height=1200)
st.plotly_chart(fig, width='stretch')

#......New cases by Region
st.subheader("New COVID-19 Cases, Deaths, and Recoveries by WHO Region")
region_summary = data.groupby('WHO Region')[['New cases', 'New deaths', 'New recovered']].sum().reset_index()
region_long = region_summary.melt(id_vars='WHO Region',
                                  value_vars=['New cases', 'New deaths', 'New recovered'],
                                  var_name='Case Type',
                                  value_name='Count')
fig = px.bar(
    region_long,
    x='WHO Region',
    y='Count',
    color='Case Type',
    barmode='group',
    text='Count'
    #title='New COVID-19 Cases, Deaths, and Recoveries by WHO Region'
)
fig.update_layout(xaxis_tickangle=-45, height=700)
st.plotly_chart(fig, width='stretch')

#.....Top 10 countries
st.subheader("Top 10 Countries by Confirmed Cases")
top10_confirmed = data.sort_values('Confirmed', ascending=False).head(10)
st.dataframe(
    top10_confirmed[['Country/Region', 'Confirmed', 'Deaths', 'Recovered', 'Active',
                     'New cases', 'New deaths', 'New recovered', 'Deaths / 100 Cases',
                     'Recovered / 100 Cases', 'Deaths / 100 Recovered',
                     'Confirmed last week', '1 week change', '1 week % increase',
                     'WHO Region']].style.background_gradient(cmap='Reds')
)

top10 = data.sort_values(by="Confirmed", ascending=False).head(10)
fig, ax = plt.subplots(figsize=(8, 5))
sns.barplot(y="Country/Region", x="Confirmed", data=top10, palette="viridis", ax=ax)
ax.set_title("Top 10 Countries by Confirmed Cases")
st.pyplot(fig)

#>>>>>>>>>>>>>>>>>>>>>>>>>4.MODELING (CLASSIFICATION)

features = ['Confirmed', 'Deaths', 'Recovered', 'Active', 'RecoveryRate', 'ActiveRatio']
target_class = 'HighDeathRate'
target_reg = 'Deaths'

X = data[features]
y_class = data[target_class]
y_reg = data[target_reg]
scaler = StandardScaler()

#....CLASSIFICATION
if model_type == "Classification":
    st.header(">> Classification Model")
    X_train, X_test, y_train, y_test = train_test_split(X, y_class, test_size=0.2, random_state=42)
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    if class_model_name == "Logistic Regression":
        model = LogisticRegression()
    elif class_model_name == "Naive Bayes":
        model = GaussianNB()
    else:
        model = SVC()

    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)

    st.write(f"### {class_model_name} Results")
    #st.text(classification_report(y_test, preds))
    #from sklearn.metrics import classification_report
    report = classification_report(y_test, preds)
    st.markdown(f"""
    ```
    {report}
    
    """)
    st.write("Accuracy:", f"{accuracy_score(y_test, preds):.4f}")

    # --- Confusion matrix heatmap ---
    cm = confusion_matrix(y_test, preds)

    # Create two axes: one for heatmap, one for colorbar
    fig, ax = plt.subplots(figsize=(3.6, 2.2))
    # Add colorbar axis beside the main plot
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)  # size and gap

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=True,
        cbar_ax=cax,   
        square=False,
        annot_kws={"size": 6, "weight": "bold"},
        xticklabels=[0, 1],
        yticklabels=[0, 1],
        ax=ax
    )

    ax.set_xlabel("Predicted Label", fontsize=6)
    ax.set_ylabel("True Label", fontsize=6)
    ax.tick_params(axis="x", labelsize=5)
    ax.tick_params(axis="y", labelsize=5)

    # Style the colorbar
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=5)

    # No tight_layout to avoid cutting colorbar
    st.pyplot(fig, clear_figure=True)

#>>>>>>>>>>>>>>>>>>>>>>>6.SUMMARY

st.header("Summary of Models and Insights")
st.markdown("""
**Classification Models Used:** Logistic Regression, Naive Bayes, Support Vector Machine (SVM)  
""")

st.success("Interactive Analysis Complete! Adjust sidebar options to explore further.")
