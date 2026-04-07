import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

st.title("🏠 House Price Prediction")

file = st.file_uploader("Upload dataset", type=["csv"])

if file:
    df = pd.read_csv(file)

    st.write("Dataset Preview")
    st.write(df.head())

    if "SalePrice" in df.columns:
        X = df.drop("SalePrice", axis=1)
        y = df["SalePrice"]

        X = pd.get_dummies(X)

        model = RandomForestRegressor()
        model.fit(X, y)

        predictions = model.predict(X)

        df["Predicted Price"] = predictions

        st.write("Predictions")
        st.write(df[["Predicted Price"]].head())
