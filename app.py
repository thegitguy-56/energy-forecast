import os
import io

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_error, mean_squared_error
from math import sqrt

def local_css(file_name: str):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("style.css")


# -----------------------------
# App config
# -----------------------------

st.set_page_config(
    page_title="Energy Consumption ARIMA Forecast",
    layout="wide"
)

st.markdown(
    "<h1 style='text-align: center; color: #222;'>Energy Consumption Forecasting (ARIMA)</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align: center; color: #555;'>Upload your energy usage CSV (time + consumption) and get a 2-month forecast with clean visuals.</p>",
    unsafe_allow_html=True
)


# -----------------------------
# Helper functions (Module 1 style)
# -----------------------------

def preprocess_uploaded_csv(uploaded_file, datetime_col, target_col,
                            freq="D", year_start="2010-01-01", year_end="2010-12-31"):
    """
    - Read CSV
    - Parse datetime_col
    - Set as index, sort
    - Resample to given freq
    - Clean missing values
    - Select 12-month window
    - Split 10 months train + 2 months test
    """
    df = pd.read_csv(uploaded_file)

    df[datetime_col] = pd.to_datetime(df[datetime_col])
    df = df.set_index(datetime_col).sort_index()

    # Resample
    df_resampled = df.resample(freq).mean()

    # Clean missing values
    df_resampled[target_col] = df_resampled[target_col].interpolate(method="time")
    df_resampled[target_col] = df_resampled[target_col].ffill()

    # Select 12-month window
    mask = (df_resampled.index >= year_start) & (df_resampled.index <= year_end)
    year_df = df_resampled.loc[mask].copy()

    # Split 10 + 2 months
    start = year_df.index.min()
    split_date = start + pd.DateOffset(months=10)

    train = year_df[year_df.index < split_date].copy()
    test = year_df[year_df.index >= split_date].copy()

    return year_df, train, test


# -----------------------------
# Helper functions (Module 2 style)
# -----------------------------

def fit_auto_arima_model(train_series: pd.Series):
    model = auto_arima(
        train_series,
        start_p=0, max_p=5,
        start_q=0, max_q=5,
        d=None,
        seasonal=False,
        stepwise=True,
        trace=False,
        error_action="ignore",
        suppress_warnings=True
    )
    return model


def evaluate_forecast(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return mae, rmse, mape


# -----------------------------
# Sidebar controls
# -----------------------------

st.sidebar.markdown("### ⚙️ Configuration")
st.sidebar.write("Upload your CSV and set the column names and date window.")
st.sidebar.markdown("---")

uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

datetime_col = st.sidebar.text_input("Datetime column name", value="Datetime")
target_col = st.sidebar.text_input("Target column (energy)", value="AEP_MW")

year_start = st.sidebar.text_input("12-month start date (YYYY-MM-DD)", value="2010-01-01")
year_end = st.sidebar.text_input("12-month end date (YYYY-MM-DD)", value="2010-12-31")
freq = st.sidebar.selectbox("Resample frequency", ["D", "H"], index=0)

if st.sidebar.button("Run Forecast") and uploaded_file is not None:
    try:
        # -----------------------------
        # 1. Preprocess (Module 1)
        # -----------------------------
        year_df, train_df, test_df = preprocess_uploaded_csv(
            uploaded_file=uploaded_file,
            datetime_col=datetime_col,
            target_col=target_col,
            freq=freq,
            year_start=year_start,
            year_end=year_end
        )

        st.success("Preprocessing completed successfully.")

        # Tabs for overview / plots
        tab1, tab2, tab3 = st.tabs(["📊 Overview", "📈 Time Series", "📉 Distribution"])

        with tab1:
            # Card 1: Cleaned dataset
            card1 = st.container()
            with card1:
                st.markdown("### Cleaned 12‑Month Dataset")
                st.dataframe(year_df.head())

            # Add a small visual separator
            st.markdown("---")

            # Card 2: Summary stats
            card2 = st.container()
            with card2:
                st.markdown("### Summary Statistics")
                st.write(year_df[target_col].describe())





        with tab2:
            st.markdown("<div class='block-card'>", unsafe_allow_html=True)
            st.subheader("Daily Energy Consumption")
            fig1, ax1 = plt.subplots(figsize=(10, 4))
            ax1.plot(year_df.index, year_df[target_col], color="#00C853")
            ax1.set_xlabel("Date")
            ax1.set_ylabel(target_col)
            ax1.grid(True)
            st.pyplot(fig1)
            st.markdown("</div>", unsafe_allow_html=True)

        with tab3:
            st.markdown("<div class='block-card'>", unsafe_allow_html=True)
            st.subheader("Consumption Distribution")
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            sns.histplot(year_df[target_col], bins=30, kde=True, color="tab:green", ax=ax2)
            st.pyplot(fig2)
            st.markdown("</div>", unsafe_allow_html=True)

        # -----------------------------
        # 2. ARIMA model (Module 2)
        # -----------------------------
        y_train = train_df[target_col]
        y_test = test_df[target_col]

        model = fit_auto_arima_model(y_train)
        order = model.order

        n_test = len(y_test)
        forecast = model.predict(n_periods=n_test)
        forecast_series = pd.Series(forecast, index=y_test.index)

        mae, rmse, mape = evaluate_forecast(y_test, forecast_series)

        st.markdown("<div class='block-card'>", unsafe_allow_html=True)
        st.subheader("Model & Metrics")
        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("MAE", f"{mae:.3f}")
        col_m2.metric("RMSE", f"{rmse:.3f}")
        col_m3.metric("MAPE", f"{mape:.2f}%")
        st.markdown(f"<p><b>Best model:</b> ARIMA{order}</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # -----------------------------
        # 3. Plot actual vs forecast
        # -----------------------------
        st.markdown("<div class='block-card'>", unsafe_allow_html=True)
        st.subheader("Forecast vs Actual (Last 2 Months)")
        fig3, ax3 = plt.subplots(figsize=(10, 4))
        ax3.plot(train_df.index, train_df[target_col], label="Train", color="tab:blue")
        ax3.plot(test_df.index, test_df[target_col], label="Test (Actual)", color="tab:green")
        ax3.plot(test_df.index, forecast_series, label="Forecast", color="tab:red")
        ax3.set_xlabel("Date")
        ax3.set_ylabel(target_col)
        ax3.grid(True)
        ax3.legend()
        st.pyplot(fig3)
        st.markdown("</div>", unsafe_allow_html=True)


        # -----------------------------
        # 4. Downloadable cleaned data
        # -----------------------------
        st.subheader("Download Cleaned Data")

        def to_csv_bytes(df):
            buffer = io.StringIO()
            df.to_csv(buffer)
            return buffer.getvalue().encode("utf-8")

        col3, col4, col5 = st.columns(3)
        with col3:
            st.download_button(
                label="Download 12‑Month Cleaned CSV",
                data=to_csv_bytes(year_df),
                file_name="cleaned_data_12months.csv",
                mime="text/csv"
            )
        with col4:
            st.download_button(
                label="Download Train (10 Months)",
                data=to_csv_bytes(train_df),
                file_name="train_data_10months.csv",
                mime="text/csv"
            )
        with col5:
            st.download_button(
                label="Download Test (2 Months)",
                data=to_csv_bytes(test_df),
                file_name="test_data_2months.csv",
                mime="text/csv"
            )

    except Exception as e:
        st.error(f"Error during processing: {e}")

elif uploaded_file is None:
    st.info("Upload a CSV file from the sidebar to begin.")
