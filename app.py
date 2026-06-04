import streamlit as st
import pandas as pd
import plotly.express as px

from analysis import (
    load_data,
    dataset_info,
    summary_statistics,
    missing_values
)

from model import train_model

st.set_page_config(
    page_title="ML Classification Dashboard",
    layout="wide"
)

st.title("Machine Learning Classification Dashboard")

uploaded_file = st.file_uploader(
    "Upload CSV File",
    type=["csv"]
)

if uploaded_file:

    df = load_data(uploaded_file)

    st.success("File Uploaded Successfully")

    st.header("Dataset Preview")

    st.dataframe(df.head())

    info = dataset_info(df)

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Rows", info["Rows"])
    c2.metric("Columns", info["Columns"])
    c3.metric("Missing Values", info["Missing Values"])
    c4.metric("Duplicates", info["Duplicates"])

    st.header("Summary Statistics")

    st.dataframe(summary_statistics(df))

    st.header("Missing Values")

    st.dataframe(
        missing_values(df).reset_index()
    )

    st.header("Target Selection")

    target_column = st.selectbox(
        "Select Target Column",
        df.columns
    )

    if st.button("Train Model"):

        with st.spinner("Training Model..."):

            (
                model,
                accuracy,
                report,
                matrix,
                feature_importance
            ) = train_model(
                df,
                target_column
            )

        st.success("Training Completed")

        st.header("Accuracy")

        st.metric(
            "Accuracy",
            f"{accuracy:.4f}"
        )

        st.header("Classification Report")

        report_df = pd.DataFrame(report).transpose()

        st.dataframe(report_df)

        st.header("Confusion Matrix")

        cm_df = pd.DataFrame(matrix)

        st.dataframe(cm_df)

        st.header("Feature Importance")

        st.dataframe(feature_importance)

        fig = px.bar(
            feature_importance,
            x="Feature",
            y="Importance",
            title="Feature Importance"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )
