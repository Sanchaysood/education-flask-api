# preprocess.py
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import joblib
import os

# Load the dataset
df = pd.read_csv("../data/2015_16_Districtwise.csv")  # go one level up to reach data

# Select columns to use
features = ['TOTPOPULAT', 'BLOCKS', 'VILLAGES', 'CLUSTERS', 'TOTCLS1G', 'TOTCLS5G']
df = df[features].dropna()

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# Create model directory if it doesn't exist (inside flask_api/)
os.makedirs("model", exist_ok=True)

# Save the scaler and models inside flask_api/model/
joblib.dump(scaler, "model/scaler.pkl")

kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_scaled)
joblib.dump(kmeans, "model/kmeans_model.pkl")

gmm = GaussianMixture(n_components=3, random_state=42)
gmm.fit(X_scaled)
joblib.dump(gmm, "model/gmm_model.pkl")

print("✅ Models saved to 'model/' folder inside flask_api.")
