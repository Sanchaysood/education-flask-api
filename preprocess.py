# preprocess.py
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import joblib
import os

# Load the dataset
df = pd.read_csv("data/2015_16_Districtwise.csv")

# Select columns to use
features = ['TOTPOPULAT', 'BLOCKS', 'VILLAGES', 'CLUSTERS', 'TOTCLS1G', 'TOTCLS5G']
df = df[features].dropna()

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# Create model directory if it doesn't exist
os.makedirs("model", exist_ok=True)

# Save the scaler
joblib.dump(scaler, "model/scaler.pkl")

# Train and save KMeans model
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_scaled)
joblib.dump(kmeans, "model/kmeans_model.pkl")

# Train and save GMM model
gmm = GaussianMixture(n_components=3, random_state=42)
gmm.fit(X_scaled)
joblib.dump(gmm, "model/gmm_model.pkl")

print("✅ Models saved to 'model/' folder.")
