import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

st.set_page_config(page_title="🎯 K-Means Playground", layout="wide")
st.title("🎯 K-Means Clustering Playground")
st.markdown("Upload a CSV file, pick features, and explore clusters interactively!")

file = st.sidebar.file_uploader("📁 Upload CSV", type=["csv"])
if file:
    df = pd.read_csv(file)
    st.markdown("## 🔍 Dataset Preview")
    st.dataframe(df.head())

    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    if len(num_cols) < 2:
        st.error("🚫 Need at least 2 numeric columns.")
    else:
        features = st.sidebar.multiselect("🧮 Select features", num_cols, default=num_cols[:2])
        k = st.sidebar.slider("🔢 Clusters (k)", 2, 10, 3)

        if len(features) >= 2:
            X_scaled = StandardScaler().fit_transform(df[features])
            df["Cluster"] = KMeans(n_clusters=k, random_state=42).fit_predict(X_scaled)

            st.markdown("## 📊 Clustered Data")
            st.dataframe(df)

            st.markdown("## 🎨 Cluster Plot")
            fig, ax = plt.subplots()
            ax.scatter(X_scaled[:, 0], X_scaled[:, 1], c=df["Cluster"], cmap="rainbow", s=80, edgecolor="black")
            ax.set_xlabel(features[0])
            ax.set_ylabel(features[1])
            ax.set_title("✨ K-Means Clustering")
            st.pyplot(fig)

            st.markdown("## 💾 Download Results")
            st.download_button("📥 Download CSV", df.to_csv(index=False).encode("utf-8"), "clustered_data.csv", "text/csv")
        else:
            st.warning("⚠️ Select at least 2 features.")
else:
    st.info("⬅️ Upload a CSV file from the sidebar to begin.")
