"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io

st.set_page_config(page_title="Data Analysis Dashboard", layout="wide")

st.title("📊 Data Analysis and Visualization Dashboard")
st.markdown("Upload your dataset and explore it interactively using Pandas, NumPy, Matplotlib, and Seaborn.")

# File uploader
uploaded_file = st.file_uploader("📂 Upload CSV File", type=["csv"])

if uploaded_file:
    # Load data
    df = pd.read_csv(uploaded_file)
    st.success("✅ Dataset Loaded Successfully!")
    
    # Dataset info
    st.subheader("📋 Dataset Overview")
    st.write(f"**Rows:** {df.shape[0]} | **Columns:** {df.shape[1]}")
    st.dataframe(df.head())

    # Sidebar for navigation
    st.sidebar.title("🔍 Navigation Menu")
    menu = st.sidebar.radio(
        "Select Option",
        ["Data Summary", "Data Visualization", "Missing Data", "Download Summary", "About"]
    )

    # Numeric and categorical separation
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()

    # 1️⃣ Data Summary
    if menu == "Data Summary":
        st.header("📈 Summary Statistics")
        st.write(df.describe())

        st.subheader("🧮 NumPy Calculations")
        for col in num_cols:
            st.write(f"**{col}** → Mean: {np.mean(df[col]):.2f}, Median: {np.median(df[col]):.2f}, Std Dev: {np.std(df[col]):.2f}")

        st.subheader("📊 Data Types")
        st.write(df.dtypes)

    # 2️⃣ Data Visualization
    elif menu == "Data Visualization":
        st.header("📉 Visualization Dashboard")
        st.sidebar.subheader("📊 Visualization Options")

        plot_type = st.sidebar.selectbox("Choose a plot type:", ["Histogram", "Bar Chart", "Box Plot", "Heatmap", "Scatter Plot", "Pairplot"])
        
        if plot_type == "Histogram":
            col = st.selectbox("Select Numeric Column", num_cols)
            bins = st.slider("Select Number of Bins", 5, 100, 20)
            plt.figure(figsize=(7,4))
            sns.histplot(df[col], bins=bins, kde=True, color="skyblue")
            plt.title(f"Histogram of {col}")
            st.pyplot(plt)

        elif plot_type == "Bar Chart":
            col = st.selectbox("Select Categorical Column", cat_cols)
            plt.figure(figsize=(7,4))
            sns.countplot(data=df, x=col, palette="coolwarm")
            plt.title(f"Bar Chart of {col}")
            st.pyplot(plt)

        elif plot_type == "Box Plot":
            col = st.selectbox("Select Numeric Column", num_cols)
            plt.figure(figsize=(7,4))
            sns.boxplot(y=df[col], color="lightgreen")
            plt.title(f"Box Plot of {col}")
            st.pyplot(plt)

        elif plot_type == "Heatmap":
            plt.figure(figsize=(10,6))
            sns.heatmap(df[num_cols].corr(), annot=True, cmap="YlGnBu", linewidths=0.3)
            plt.title("Correlation Heatmap")
            st.pyplot(plt)

        elif plot_type == "Scatter Plot":
            col1 = st.selectbox("X-axis", num_cols)
            col2 = st.selectbox("Y-axis", num_cols)
            plt.figure(figsize=(7,4))
            sns.scatterplot(x=df[col1], y=df[col2], hue=df[cat_cols[0]] if cat_cols else None, palette="viridis")
            plt.title(f"Scatter Plot: {col1} vs {col2}")
            st.pyplot(plt)

        elif plot_type == "Pairplot":
            selected_cols = st.multiselect("Select Columns for Pairplot", num_cols)
            if len(selected_cols) > 1:
                sns.pairplot(df[selected_cols])
                st.pyplot(plt)
            else:
                st.warning("Please select at least two numeric columns for Pairplot.")

    # 3️⃣ Missing Data
    elif menu == "Missing Data":
        st.header("🧩 Missing Data Analysis")
        missing = df.isnull().sum().reset_index()
        missing.columns = ["Column", "Missing Values"]
        missing["% Missing"] = (missing["Missing Values"] / len(df)) * 100
        st.dataframe(missing)

        st.subheader("📉 Visualizing Missing Data")
        plt.figure(figsize=(10,5))
        sns.heatmap(df.isnull(), cbar=False, cmap="Blues")
        st.pyplot(plt)

        if missing["Missing Values"].sum() > 0:
            option = st.radio("Handle Missing Data:", ["Do Nothing", "Drop Rows", "Fill with Mean"])
            if option == "Drop Rows":
                df.dropna(inplace=True)
                st.success("✅ Missing rows dropped!")
            elif option == "Fill with Mean":
                df.fillna(df.mean(numeric_only=True), inplace=True)
                st.success("✅ Missing values filled with mean!")

    # 4️⃣ Download Summary
    elif menu == "Download Summary":
        st.header("📥 Download Data Summary")
        buffer = io.StringIO()
        df.describe().to_csv(buffer)
        st.download_button(
            label="Download Summary as CSV",
            data=buffer.getvalue(),
            file_name="data_summary.csv",
            mime="text/csv"
        )
        st.success("✅ Click the button above to download your summary!")

    # 5️⃣ About Section
    elif menu == "About":
        st.header("ℹ️ About this Dashboard")
        st.markdown(
        **Data Analysis and Visualization Dashboard**

        - **Developed with:** Python, Pandas, NumPy, Matplotlib, Seaborn, and Streamlit  
        - **Features:**
          - Upload any CSV dataset  
          - Explore descriptive statistics  
          - Visualize data distributions and relationships  
          - Detect and handle missing data  
          - Download summary reports  
        - **Ideal for:** Data Analysis, Machine Learning, and Statistics courses  
        )
        st.info("💡 Tip: You can deploy this dashboard to Streamlit Cloud or Render for free.")

else:
    st.info("👆 Please upload a CSV file to start exploring your data.")
"""

# machine_learning_model_explorer.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, silhouette_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.cluster import AgglomerativeClustering, DBSCAN
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="ML Model Explorer", layout="wide")

# -------------------- Title --------------------
st.title("🤖 Machine Learning Model Explorer")
st.markdown("Explore Supervised and Unsupervised Machine Learning Models interactively using Streamlit.")

# -------------------- Sidebar --------------------
st.sidebar.header("⚙️ Model Configuration")
learning_type = st.sidebar.selectbox("Select Learning Type", ["Supervised", "Unsupervised"])

# -------------------- File Upload --------------------
uploaded_file = st.sidebar.file_uploader("Upload your dataset (CSV format)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head())

    # Handle missing values
    if df.isnull().sum().sum() > 0:
        st.warning("Missing values found. They will be filled automatically.")
        df = df.fillna(df.mode().iloc[0])

    # Encode categorical variables
    label_encoders = {}
    for col in df.select_dtypes(include='object').columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    # Scale data
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

    # -------------------- Supervised Learning --------------------
    if learning_type == "Supervised":
        target_col = st.sidebar.selectbox("Select Target Column", df.columns)

        X = df_scaled.drop(columns=[target_col])
        y = df[target_col]

        test_size = st.sidebar.slider("Test Size (for train/test split)", 0.1, 0.5, 0.2)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        model_name = st.sidebar.selectbox("Select Algorithm", ["Decision Tree", "Random Forest", "SVM"])

        # Model selection
        if model_name == "Decision Tree":
            max_depth = st.sidebar.slider("Max Depth", 1, 20, 5)
            model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)

        elif model_name == "Random Forest":
            n_estimators = st.sidebar.slider("Number of Trees", 10, 200, 100)
            model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)

        else:  # SVM
            kernel = st.sidebar.selectbox("Kernel", ["linear", "rbf", "poly"])
            model = SVC(kernel=kernel)

        if st.sidebar.button("Train Model"):
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            st.success(f"✅ Model trained successfully! Accuracy: **{acc:.2f}**")

            # Confusion Matrix
            cm = confusion_matrix(y_test, y_pred)
            st.subheader("🔢 Confusion Matrix")
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
            st.pyplot(fig)

            # Classification Report
            st.subheader("📋 Classification Report")
            st.text(classification_report(y_test, y_pred))

    # -------------------- Unsupervised Learning --------------------
    else:
        model_name = st.sidebar.selectbox("Select Algorithm", ["Agglomerative Clustering", "DBSCAN"])
        X = df_scaled.copy()

        if model_name == "Agglomerative Clustering":
            n_clusters = st.sidebar.slider("Number of Clusters", 2, 10, 3)
            model = AgglomerativeClustering(n_clusters=n_clusters)
        else:
            eps = st.sidebar.slider("Epsilon (eps)", 0.1, 5.0, 0.5)
            min_samples = st.sidebar.slider("Min Samples", 2, 20, 5)
            model = DBSCAN(eps=eps, min_samples=min_samples)

        if st.sidebar.button("Run Clustering"):
            labels = model.fit_predict(X)
            df['Cluster'] = labels
            st.success("✅ Clustering completed successfully!")

            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            st.write(f"Detected Clusters: **{n_clusters}**")

            # Silhouette Score (if valid)
            if len(set(labels)) > 1 and -1 not in labels:
                score = silhouette_score(X, labels)
                st.write(f"Silhouette Score: **{score:.2f}**")

            # Cluster Plot
            st.subheader("📈 Cluster Visualization")
            if X.shape[1] >= 2:
                fig, ax = plt.subplots()
                plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=labels, cmap="viridis", s=50)
                plt.xlabel(X.columns[0])
                plt.ylabel(X.columns[1])
                st.pyplot(fig)
            else:
                st.warning("Dataset has less than 2 features — cannot plot clusters.")

else:
    st.info("👆 Please upload a CSV file to get started.")

# -------------------- Footer --------------------
st.markdown("---")
st.markdown("**Developed as a Mini Project for Machine Learning Course** 🧠")

