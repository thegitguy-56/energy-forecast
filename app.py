import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.statespace.sarimax import SARIMAX
from math import sqrt

# -----------------------------
# Load local CSS (optional)
# -----------------------------
def local_css(file_name: str):
    try:
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except Exception:
        # silently ignore if style file not found
        pass


local_css("style.css")

# -----------------------------
# App config
# -----------------------------
st.set_page_config(
    page_title="Energy Consumption Forecast",
    layout="wide"
)

st.markdown("<h1 style='text-align: center; color: #222;'>Energy Consumption Forecasting</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #555;'>Upload a CSV (datetime + energy) and get a 2-month forecast with SARIMAX and Random Forest.</p>", unsafe_allow_html=True)

# -----------------------------
# Helper functions
# -----------------------------
def preprocess_uploaded_csv(uploaded_file, datetime_col, target_col,
                            freq="D", year_start="2010-01-01", year_end="2010-12-31"):
    df = pd.read_csv(uploaded_file)
    df[datetime_col] = pd.to_datetime(df[datetime_col])
    df = df.set_index(datetime_col).sort_index()

    # Resample and aggregate
    df_resampled = df.resample(freq).mean()

    # Interpolate / forward fill target
    df_resampled[target_col] = df_resampled[target_col].interpolate(method="time")
    df_resampled[target_col] = df_resampled[target_col].ffill()

    # Select 12-month window and split 10 + 2 months
    mask = (df_resampled.index >= year_start) & (df_resampled.index <= year_end)
    year_df = df_resampled.loc[mask].copy()

    start = year_df.index.min()
    split_date = start + pd.DateOffset(months=10)

    train = year_df[year_df.index < split_date].copy()
    test = year_df[year_df.index >= split_date].copy()

    return year_df, train, test


def evaluate_forecast(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = sqrt(mean_squared_error(y_true, y_pred))
    # avoid division by zero in mape
    nonzero = y_true.replace(0, np.finfo(float).eps)
    mape = np.mean(np.abs((nonzero - y_pred) / nonzero)) * 100
    return mae, rmse, mape


def make_lag_features(series: pd.Series, n_lags: int = 7):
    df = pd.DataFrame({"y": series})
    for lag in range(1, n_lags + 1):
        df[f"lag_{lag}"] = df["y"].shift(lag)
    df = df.dropna()
    X = df[[f"lag_{lag}" for lag in range(1, n_lags + 1)]]
    y = df["y"]
    return X, y

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
run_btn = st.sidebar.button("Run Forecast")

# -----------------------------
# Main logic
# -----------------------------
if run_btn and uploaded_file is not None:
    try:
        # 1. Preprocess
        year_df, train_df, test_df = preprocess_uploaded_csv(
            uploaded_file=uploaded_file,
            datetime_col=datetime_col,
            target_col=target_col,
            freq=freq,
            year_start=year_start,
            year_end=year_end
        )

        # Ensure numeric targets
        train_df[target_col] = pd.to_numeric(train_df[target_col].astype(str).str.replace(",", "").str.strip(), errors="coerce")
        test_df[target_col] = pd.to_numeric(test_df[target_col].astype(str).str.replace(",", "").str.strip(), errors="coerce")
        train_df = train_df.dropna(subset=[target_col])
        test_df = test_df.dropna(subset=[target_col])

        y_train = train_df[target_col].astype(float)
        y_test = test_df[target_col].astype(float)
        n_test = len(y_test)

        # -----------------------------
        # SARIMAX with exogenous lag features
        # -----------------------------
        def make_sarimax_features(series: pd.Series):
            df = pd.DataFrame({"y": series})
            df["lag_1"] = df["y"].shift(1)
            df["lag_7"] = df["y"].shift(7)
            df["rolling_7"] = df["y"].rolling(7).mean()
            return df

        train_exo = make_sarimax_features(y_train).dropna()
        combined = pd.concat([y_train, y_test])
        combined_exo = make_sarimax_features(combined)
        test_exo = combined_exo.loc[y_test.index].copy()

        y_train_aligned = y_train.loc[train_exo.index].copy()

        sarimax_model = SARIMAX(
            y_train_aligned,
            exog=train_exo[["lag_1", "lag_7", "rolling_7"]],
            order=(2, 0, 2),
            seasonal_order=(1, 1, 1, 7),
            enforce_stationarity=False,
            enforce_invertibility=False
        ).fit(disp=False)

        sarimax_pred = sarimax_model.forecast(
            steps=n_test,
            exog=test_exo[["lag_1", "lag_7", "rolling_7"]]
        )
        sarimax_forecast = pd.Series(np.asarray(sarimax_pred, dtype=float), index=y_test.index)

        mae_sx, rmse_sx, mape_sx = evaluate_forecast(y_test, sarimax_forecast)

        # -----------------------------
        # Random Forest baseline
        # -----------------------------
        X_train_rf, y_train_rf = make_lag_features(y_train, n_lags=7)
        rf = RandomForestRegressor(n_estimators=200, random_state=42)
        rf.fit(X_train_rf, y_train_rf)

        combined_rf = pd.concat([y_train, y_test])
        X_all_rf, y_all_rf = make_lag_features(combined_rf, n_lags=7)

        # align test features
        common_index = X_all_rf.index.intersection(y_test.index)
        if len(common_index) >= n_test:
            X_test_rf = X_all_rf.loc[common_index].sort_index()
            X_test_rf = X_test_rf.reindex(y_test.index)
        else:
            X_test_rf = X_all_rf.tail(n_test)
            if not X_test_rf.index.equals(y_test.index):
                rf_index = X_test_rf.index
                y_test_aligned = y_test.reindex(rf_index)
            else:
                y_test_aligned = y_test

        rf_pred = rf.predict(X_test_rf)
        rf_forecast = pd.Series(rf_pred, index=X_test_rf.index)

        if 'y_test_aligned' in locals():
            eval_y_test = y_test_aligned
        else:
            eval_y_test = y_test.reindex(rf_forecast.index)

        mae_rf, rmse_rf, mape_rf = evaluate_forecast(eval_y_test, rf_forecast)

      
        # -----------------------------
        # UI: Overview + Model-specific Tabs + Project Summary
        # -----------------------------
        tab_overview, tab_ts, tab_dist, tab_sarimax, tab_rf, tab_summary = st.tabs(
            ["📊 Overview", "📈 Time Series", "📉 Distribution", "🧮 SARIMAX", "🌲 Random Forest", "📄 Project Summary"]
        )

        with tab_overview:
            st.markdown("### Cleaned 12-Month Dataset")
            st.dataframe(year_df.head())
            st.markdown("---")
            st.markdown("### Summary Statistics")
            st.write(year_df[target_col].describe())

        with tab_ts:
            st.markdown("### Daily Energy Consumption")
            fig1, ax1 = plt.subplots(figsize=(10, 4))
            ax1.plot(year_df.index, year_df[target_col], color="#00C853")
            ax1.set_xlabel("Date")
            ax1.set_ylabel(target_col)
            ax1.grid(True)
            st.pyplot(fig1)

        with tab_dist:
            st.markdown("### Consumption Distribution")
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            sns.histplot(year_df[target_col], bins=30, kde=True, ax=ax2)
            st.pyplot(fig2)

        with tab_sarimax:
            st.markdown("### SARIMAX — Metrics & Forecast")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("MAE", f"{mae_sx:.3f}")
            with c2:
                st.metric("RMSE", f"{rmse_sx:.3f}")
            with c3:
                st.metric("MAPE", f"{mape_sx:.2f}%")

            st.markdown("#### SARIMAX Forecast vs Actual (Test Period)")
            fig_sx, ax_sx = plt.subplots(figsize=(10, 4))
            ax_sx.plot(train_df.index, train_df[target_col], label="Train", color="tab:blue", linewidth=1)
            ax_sx.plot(test_df.index, test_df[target_col], label="Actual", color="tab:green", linewidth=1)
            ax_sx.plot(sarimax_forecast.index, sarimax_forecast, label="SARIMAX Forecast", color="tab:red", linewidth=2)
            ax_sx.set_xlabel("Date")
            ax_sx.set_ylabel(target_col)
            ax_sx.grid(True)
            ax_sx.legend()
            st.pyplot(fig_sx)

            st.markdown("#### Sample: Actual vs SARIMAX")
            sample_sx = pd.DataFrame({"Actual": y_test, "SARIMAX": sarimax_forecast}).head(20)
            st.dataframe(sample_sx.style.format({"Actual": "{:.2f}", "SARIMAX": "{:.2f}"}))

        with tab_rf:
            st.markdown("### Random Forest — Metrics & Forecast")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("MAE", f"{mae_rf:.3f}")
            with c2:
                st.metric("RMSE", f"{rmse_rf:.3f}")
            with c3:
                st.metric("MAPE", f"{mape_rf:.2f}%")

            st.markdown("#### Random Forest Forecast vs Actual (Test Period)")
            fig_rf, ax_rf = plt.subplots(figsize=(10, 4))
            ax_rf.plot(train_df.index, train_df[target_col], label="Train", color="tab:blue", linewidth=1)
            ax_rf.plot(test_df.index, test_df[target_col], label="Actual", color="tab:green", linewidth=1)
            ax_rf.plot(rf_forecast.index, rf_forecast, label="Random Forest Forecast", color="tab:orange", linestyle="--", linewidth=2)
            ax_rf.set_xlabel("Date")
            ax_rf.set_ylabel(target_col)
            ax_rf.grid(True)
            ax_rf.legend()
            st.pyplot(fig_rf)

            st.markdown("#### Sample: Actual vs Random Forest")
            # use eval_y_test (from RF alignment) if present; otherwise show y_test
            rf_table_y = eval_y_test if 'eval_y_test' in locals() else y_test
            sample_rf = pd.DataFrame({"Actual": rf_table_y, "RandomForest": rf_forecast}).head(20)
            st.dataframe(sample_rf.style.format({"Actual": "{:.2f}", "RandomForest": "{:.2f}"}))

        with tab_summary:
            # Build a nicely formatted project summary (markdown)
            project_title = "Energy Consumption Forecasting"
            dataset_range = f"{year_start} to {year_end}"
            model_summary = (
                f"- SARIMAX (weekly seasonality): MAE={mae_sx:.2f}, RMSE={rmse_sx:.2f}, MAPE={mape_sx:.2f}%\n"
                f"- Random Forest (lags 1..7): MAE={mae_rf:.2f}, RMSE={rmse_rf:.2f}, MAPE={mape_rf:.2f}%\n"
            )

            summary_md = f"""
        # {project_title}

        **Dataset:** `{datetime_col}` (index) and `{target_col}` (target)  
        **Date window used:** {dataset_range}  
        **Resample frequency:** {freq}

        ---

        ## Abstract
        Short project: forecast short-term (2-month) energy consumption using a classical time-series model (SARIMAX) and a machine-learning baseline (Random Forest) using lag features.

        ## Objectives
        - Forecast energy consumption for the next 2 months.
        - Compare classical time-series methods with machine learning approaches.
        - Provide visualizations and downloadable cleaned data for reporting.

        ## Methodology
        1. Preprocess: parse datetime, resample (user-selected freq), interpolate missing values, select 12-month window and split into 10 months train + 2 months test.
        2. SARIMAX: build lag-based exogenous features (lag_1, lag_7, rolling_7) and fit a seasonal ARIMAX model capturing weekly seasonality.
        3. Random Forest: fit a lag-feature based Random Forest regressor (lags 1..7) as baseline.
        4. Evaluate both models on the 2-month test window using MAE, RMSE, and MAPE.

        ## Key Results
        {model_summary}

        ## Conclusions
        - SARIMAX captures seasonality and trend; it produces smooth forecasts and performs well on seasonal patterns.
        - Random Forest captures spikes and short-term nonlinear effects better due to lag features.
        - Use ensemble/hybrid approaches or additional exogenous variables (temperature, holidays) for further improvement.

        ## Challenges & Limitations
        - ARIMA-family models smooth long horizons and may underpredict spikes.
        - Model performance depends on training window selection and data quality.
        - No external regressors (weather, events) included in the current pipeline.

        ## Future Work
        - Add exogenous data (temperature, holidays) to improve accuracy.
        - Experiment with ensemble/hybrid models (SARIMAX + RF blending).
        - Add cross-validation and hyper-parameter search for RF and SARIMAX.
        - Consider deep-learning sequence models (LSTM/Transformer) for complex patterns.

        ---

        ### Notes
        You can download this summary as a markdown file below.
        """

            st.markdown(summary_md)

            # Provide download button for markdown summary
            import io as _io
            buf = _io.StringIO()
            buf.write(summary_md)
            data_bytes = buf.getvalue().encode("utf-8")
            st.download_button(
                label="📥 Download Project Summary (Markdown)",
                data=data_bytes,
                file_name="project_summary.md",
                mime="text/markdown"
            )

        # Metrics summary
        st.subheader("Model Metrics (Test Period)")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**SARIMAX**")
            st.metric("MAE", f"{mae_sx:.3f}")
            st.metric("RMSE", f"{rmse_sx:.3f}")
            st.metric("MAPE", f"{mape_sx:.2f}%")
        with col2:
            st.markdown("**Random Forest**")
            st.metric("MAE", f"{mae_rf:.3f}")
            st.metric("RMSE", f"{rmse_rf:.3f}")
            st.metric("MAPE", f"{mape_rf:.2f}%")

        # Forecast vs Actual (last 2 months)
        st.subheader("Forecast vs Actual (Test Period)")
        fig_f, ax_f = plt.subplots(figsize=(10, 4))
        ax_f.plot(train_df.index, train_df[target_col], label="Train", color="tab:blue", linewidth=1)
        ax_f.plot(test_df.index, test_df[target_col], label="Test (Actual)", color="tab:green", linewidth=1)
        ax_f.plot(sarimax_forecast.index, sarimax_forecast, label="SARIMAX Forecast", color="tab:red", linewidth=2)
        ax_f.plot(rf_forecast.index, rf_forecast, label="Random Forest Forecast", color="tab:orange", linestyle="--")
        ax_f.set_xlabel("Date")
        ax_f.set_ylabel(target_col)
        ax_f.grid(True)
        ax_f.legend()
        st.pyplot(fig_f)

        # Comparison table
        comp_df = pd.DataFrame({
            "Actual": y_test,
            "SARIMAX": sarimax_forecast,
        }).join(rf_forecast.rename("RandomForest"), how="left")
        st.markdown("### Actual vs Forecast (sample)")
        st.dataframe(comp_df.head(16).style.format({"Actual": "{:.2f}", "SARIMAX": "{:.2f}", "RandomForest": "{:.2f}"}))

        # Download cleaned data
        st.subheader("Download Cleaned Data")
        def to_csv_bytes(df):
            buffer = io.StringIO()
            df.to_csv(buffer)
            return buffer.getvalue().encode("utf-8")

        c1, c2, c3 = st.columns(3)
        with c1:
            st.download_button("Download 12-Month Cleaned CSV", data=to_csv_bytes(year_df),
                               file_name="cleaned_data_12months.csv", mime="text/csv")
        with c2:
            st.download_button("Download Train (10 Months)", data=to_csv_bytes(train_df),
                               file_name="train_data_10months.csv", mime="text/csv")
        with c3:
            st.download_button("Download Test (2 Months)", data=to_csv_bytes(test_df),
                               file_name="test_data_2months.csv", mime="text/csv")

    except Exception as e:
        st.error(f"Error during processing: {e}")

elif uploaded_file is None:
    st.info("Upload a CSV file from the sidebar to begin.")
