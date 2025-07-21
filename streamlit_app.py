import streamlit as st
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.decomposition import PCA

# Page setup
st.set_page_config(page_title="Education Clustering", layout="wide")
st.title("📊 District Education Infrastructure Clustering")

st.markdown("""
<style>
    .main {
        background-color: #0e1117;
        color: #ffffff;
    }
    .stSlider > div[data-baseweb="slider"] {
        background-color: #1c1f26;
        border-radius: 0.5rem;
        padding: 0.5rem;
    }
    .stSelectbox > div[data-baseweb="select"] {
        background-color: #1c1f26;
        border-radius: 0.5rem;
        color: #ffffff;
    }
    .stButton > button {
        background-color: #ff4b4b;
        color: white;
        border-radius: 0.5rem;
        padding: 0.5rem 1rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("Use the sliders below to input district-level education and population data. Choose a clustering model to predict which cluster your district belongs to.")

# --- Load dataset ---
df_data = pd.read_csv("data/2015_16_Districtwise.csv")
df_data.columns = df_data.columns.str.strip()
columns = ['TOTPOPULAT', 'BLOCKS', 'VILLAGES', 'CLUSTERS', 'TOTCLS1G', 'TOTCLS5G']
df_data = df_data.dropna(subset=columns)

# Function to round min/max values for sliders
def round_min_max(col, min_val, max_val):
    if col == 'TOTPOPULAT':
        return (round(min_val, -3), round(max_val, -3))
    elif col == 'BLOCKS':
        return (round(min_val), round(max_val))
    elif col in ['VILLAGES', 'CLUSTERS']:
        return (round(min_val, -1), round(max_val, -1))
    elif col in ['TOTCLS1G', 'TOTCLS5G']:
        return (round(min_val, -2), round(max_val, -2))
    else:
        return (min_val, max_val)

# Prepare rounded min-max dictionary
min_max = {}
for col in columns:
    min_val = int(df_data[col].min())
    max_val = int(df_data[col].max())
    min_max[col] = round_min_max(col, min_val, max_val)

# Set default values to min
default_values = {col: min_max[col][0] for col in columns}

# --- User Input Form ---
with st.form("input_form"):
    st.markdown("### 🔧 Input Your District's Details")
    col1, col2 = st.columns(2)
    with col1:
        population = st.slider("📈 Total Population", min_value=min_max['TOTPOPULAT'][0], max_value=min_max['TOTPOPULAT'][1], value=default_values['TOTPOPULAT'])
        blocks = st.slider("🏢 Administrative Blocks", min_value=min_max['BLOCKS'][0], max_value=min_max['BLOCKS'][1], value=default_values['BLOCKS'])
        villages = st.slider("🏘️ Villages", min_value=min_max['VILLAGES'][0], max_value=min_max['VILLAGES'][1], value=default_values['VILLAGES'])
    with col2:
        clusters = st.slider("🏫 School Clusters", min_value=min_max['CLUSTERS'][0], max_value=min_max['CLUSTERS'][1], value=default_values['CLUSTERS'])
        class1 = st.slider("👶 Class 1 Enrollments", min_value=min_max['TOTCLS1G'][0], max_value=min_max['TOTCLS1G'][1], value=default_values['TOTCLS1G'])
        class5 = st.slider("👦 Class 5 Enrollments", min_value=min_max['TOTCLS5G'][0], max_value=min_max['TOTCLS5G'][1], value=default_values['TOTCLS5G'])

    selected_model = st.selectbox("🤖 Select Clustering Model", ["KMeans", "GMM"])
    submitted = st.form_submit_button("🔍 Predict")

# --- On Form Submission ---
if submitted:
    features = [population, blocks, villages, clusters, class1, class5]

    try:
        response = requests.post("https://education-flask-api-2.onrender.com/predict", json={"features": features})

        if response.ok:
            result = response.json()
            cluster_id = result['kmeans_cluster'] if selected_model == "KMeans" else result['gmm_cluster']

            st.markdown("## 🔍 Prediction Result")
            st.success(f"✅ {selected_model} Cluster: {cluster_id}")

            # 🎯 Application Purpose
            st.markdown("### 🎯 Final Recommendation")
            if selected_model == "KMeans":
                if cluster_id == 0:
                    st.info("🟢 **Your district is well-developed in terms of education infrastructure. Keep maintaining the current growth.**")
                elif cluster_id == 1:
                    st.warning("🟡 **Your district is growing. Consider increasing the number of schools or improving enrollment.**")
                else:
                    st.error("🔴 **Your district needs improvement. Focus on increasing the number of school clusters and improving access to education.**")
            else:
                if cluster_id == 0:
                    st.error("🔴 **Your district shows urgent need for infrastructure support. Prioritize adding more clusters and balancing enrollment.**")
                elif cluster_id == 1:
                    st.warning("🟡 **Your district is moderately developed. Focused improvements could lead to higher development.**")
                else:
                    st.info("🟢 **Your district has balanced infrastructure. Monitor progress and maintain growth.**")

            # Load models from correct path (because this script is outside flask_api/)
            scaler = joblib.load("flask_api/model/scaler.pkl")
            kmeans = joblib.load("flask_api/model/kmeans_model.pkl")
            gmm = joblib.load("flask_api/model/gmm_model.pkl")

            df = df_data.copy()
            X_scaled = scaler.transform(df[columns])
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_scaled)

            model = kmeans if selected_model == "KMeans" else gmm
            df["Cluster"] = model.predict(X_scaled)
            cmap = 'Set1' if selected_model == "KMeans" else 'Set2'
            title = f"{selected_model} Clustering"

            # Display three plots side by side
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown(f"### 📈 {title}")
                fig, ax = plt.subplots(figsize=(4, 3))
                ax.scatter(X_pca[:, 0], X_pca[:, 1], c=df["Cluster"], cmap=cmap)
                ax.set_title(title, fontsize=10)
                user_scaled = scaler.transform([features])
                user_pca = pca.transform(user_scaled)
                ax.scatter(user_pca[0, 0], user_pca[0, 1], color='red', s=50, label="Your District", marker='*')
                ax.legend()
                st.pyplot(fig)

            with col2:
                st.markdown("### 📊 Cluster Size Distribution")
                cluster_counts = df['Cluster'].value_counts().sort_index()
                fig2, ax2 = plt.subplots(figsize=(4, 3))
                ax2.bar(cluster_counts.index.astype(str), cluster_counts.values, color='skyblue')
                ax2.set_title("Cluster Size")
                st.pyplot(fig2)

            with col3:
                st.markdown("### 📦 Feature Distribution: Total Population")
                fig3, ax3 = plt.subplots(figsize=(4, 3))
                sns.boxplot(x="Cluster", y="TOTPOPULAT", data=df, ax=ax3, palette=cmap)
                ax3.set_title("Total Population by Cluster")
                st.pyplot(fig3)

        else:
            st.error("❌ Failed to get a response from the Flask server.")

    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
