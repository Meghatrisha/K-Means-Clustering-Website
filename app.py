import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# --- Streamlit UI ---
st.title("K-Means Clustering App")
st.write("Upload a CSV dataset and choose the number of clusters to visualize K-Means clustering.")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### Preview of Dataset")
    st.dataframe(df.head())

    # Let user choose numeric columns only
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    if len(numeric_cols) < 2:
        st.error("Dataset must contain at least 2 numeric columns for clustering.")
    else:
        selected_features = st.multiselect(
            "Select features for clustering (min 2 columns)", 
            numeric_cols, 
            default=numeric_cols[:2]
        )

        if len(selected_features) >= 2:
            X = df[selected_features]

            k = st.slider("Select Number of Clusters (k)", min_value=2, max_value=10, value=3)

            # Scale data
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # KMeans model
            kmeans = KMeans(n_clusters=k, random_state=42)
            clusters = kmeans.fit_predict(X_scaled)

            df["Cluster"] = clusters
            st.write("### Clustered Data")
            st.dataframe(df)

            # Plot clusters (first 2 features)
            fig, ax = plt.subplots()
            scatter = ax.scatter(
                X_scaled[:, 0],
                X_scaled[:, 1],
                c=clusters
            )

            ax.set_xlabel(selected_features[0])
            ax.set_ylabel(selected_features[1])
            ax.set_title("K-Means Clustering Visualization")
            st.pyplot(fig)

            # Download clustered data
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download Clustered Data",
                data=csv,
                file_name="clustered_data.csv",
                mime="text/csv"
            )
        else:
            st.warning("Please select at least 2 numeric features for clustering.")
else:
    st.info("⬆️ Upload a CSV file to begin.")
