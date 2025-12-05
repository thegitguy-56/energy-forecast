import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error
from math import sqrt

from pmdarima import auto_arima
from statsmodels.tsa.arima.model import ARIMA


# -----------------------------
# Configuration
# -----------------------------

DATA_DIR = os.path.join("..", "data", "processed")
PLOTS_DIR = os.path.join("..", "outputs", "plots")

TRAIN_FILE = "train_data_10months.csv"
TEST_FILE = "test_data_2months.csv"

TARGET_COL = "AEP_MW"  # same as Module 1


def ensure_directories():
    os.makedirs(PLOTS_DIR, exist_ok=True)


# -----------------------------
# Step 1: Load processed data
# -----------------------------

def load_train_test():
    train_path = os.path.join(DATA_DIR, TRAIN_FILE)
    test_path = os.path.join(DATA_DIR, TEST_FILE)

    print(f"Loading train data from: {train_path}")
    print(f"Loading test data from: {test_path}")

    train = pd.read_csv(train_path, parse_dates=True, index_col=0)
    test = pd.read_csv(test_path, parse_dates=True, index_col=0)

    print("Train shape:", train.shape)
    print("Test shape:", test.shape)

    return train, test


# -----------------------------
# Step 2: Fit Auto-ARIMA model
# -----------------------------

def fit_auto_arima_model(train_series: pd.Series):
    """
    Use auto_arima to automatically choose (p, d, q) parameters.
    We only specify ranges; it fills the 'empty' best values.
    """
    print("\nFitting auto-ARIMA model. This may take a little time...")

    # seasonal=False because we are using daily data for one year;
    # if you want seasonal effects (e.g. weekly), you can set seasonal=True, m=7.
    model = auto_arima(
        train_series,
        start_p=0, max_p=5,
        start_q=0, max_q=5,
        d=None,           # let auto_arima find the differencing
        seasonal=False,   # change to True and set m for seasonal data
        stepwise=True,
        trace=True,
        error_action="ignore",
        suppress_warnings=True
    )

    print("\nBest ARIMA order found:", model.order)
    return model


# -----------------------------
# Step 3: Forecast on test period
# -----------------------------

def forecast_with_model(auto_model, n_periods: int):
    """
    Forecast n_periods into the future using the fitted auto_arima model.
    """
    print(f"\nForecasting next {n_periods} periods...")
    forecast = auto_model.predict(n_periods=n_periods)
    return forecast


# -----------------------------
# Step 4: Evaluate performance
# -----------------------------

def evaluate_forecast(y_true, y_pred):
    """
    Calculate MAE, RMSE, MAPE between true and predicted values.
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    print("\nEvaluation Metrics:")
    print(f"MAE  : {mae:.3f}")
    print(f"RMSE : {rmse:.3f}")
    print(f"MAPE : {mape:.2f}%")

    return mae, rmse, mape


# -----------------------------
# Step 5: Plot actual vs predicted
# -----------------------------

def plot_actual_vs_predicted(train, test, forecast, title, file_name):
    """
    Plot training data, test data (actual) and forecast (predictions).
    """
    plt.figure(figsize=(12, 5))

    # Plot train
    plt.plot(train.index, train[TARGET_COL], label="Train", color="tab:blue")

    # Plot test actual
    plt.plot(test.index, test[TARGET_COL], label="Test (Actual)", color="tab:green")

    # Plot forecast aligned to test index
    plt.plot(test.index, forecast, label="Forecast", color="tab:red")

    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel(TARGET_COL)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    save_path = os.path.join(PLOTS_DIR, file_name)
    plt.savefig(save_path)
    plt.close()
    print(f"Saved plot: {save_path}")


# -----------------------------
# Optional: Refit using statsmodels
# -----------------------------

def refit_statsmodels_arima(train_series: pd.Series, order):
    """
    Optional step: fit a statsmodels ARIMA with the order found by auto_arima,
    for more diagnostics or advanced options later.
    """
    print("\nRefitting ARIMA with statsmodels using order:", order)
    model = ARIMA(train_series, order=order)
    result = model.fit()
    print(result.summary())
    return result


# -----------------------------
# Main pipeline for Module 2
# -----------------------------

def run_module2():
    ensure_directories()

    # 1. Load data
    train_df, test_df = load_train_test()

    # Extract series
    y_train = train_df[TARGET_COL]
    y_test = test_df[TARGET_COL]

    # 2. Fit Auto-ARIMA
    auto_model = fit_auto_arima_model(y_train)

    # 3. Forecast length = length of test set
    n_test = len(y_test)
    forecast = forecast_with_model(auto_model, n_periods=n_test)

    # Align forecast as a Series with same index as test
    forecast_series = pd.Series(forecast, index=y_test.index)

    # 4. Evaluate
    mae, rmse, mape = evaluate_forecast(y_test, forecast_series)

    # 5. Plot actual vs predicted
    plot_actual_vs_predicted(
        train=train_df,
        test=test_df,
        forecast=forecast_series,
        title="ARIMA Forecast vs Actual (Daily Energy Consumption)",
        file_name="arima_forecast_vs_actual.png"
    )

    # 6. Optional: refit statsmodels ARIMA using discovered order
    # This can be useful for diagnostics, residual analysis, etc.
    order = auto_model.order
    _ = refit_statsmodels_arima(y_train, order=order)

    print("\nModule 2 ARIMA modeling completed successfully.")


if __name__ == "__main__":
    run_module2()
