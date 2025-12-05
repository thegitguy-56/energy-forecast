import os
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# -----------------------------
# Configuration
# -----------------------------

# Relative paths (change if your structure is different)
DATA_DIR = os.path.join("..", "data")
OUTPUT_DATA_DIR = os.path.join("..", "data", "processed")
PLOTS_DIR = os.path.join("..", "outputs", "plots")

# Input file name from Kaggle (placed in data/ folder)
INPUT_FILE = "AEP_hourly.csv"  # from Hourly Energy Consumption dataset

# Column names in AEP_hourly.csv (adjust if different)
DATETIME_COL = "Datetime"
TARGET_COL = "AEP_MW"

# Desired time frequency for ARIMA (daily)
RESAMPLE_FREQ = "D"

# Year to use for the 12-month window (adjust if needed)
YEAR_START = "2010-01-01"
YEAR_END = "2010-12-31"

# Number of months for training (out of 12)
TRAIN_MONTHS = 10


# -----------------------------
# Utility: ensure folders exist
# -----------------------------

def ensure_directories():
    os.makedirs(OUTPUT_DATA_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)


# -----------------------------
# Step 1: Load raw data
# -----------------------------

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load the raw CSV file, parse datetime, set index, and sort by time.
    """
    print(f"Loading data from: {file_path}")
    df = pd.read_csv(file_path)

    # Convert datetime column
    df[DATETIME_COL] = pd.to_datetime(df[DATETIME_COL])

    # Set datetime as index and sort
    df = df.set_index(DATETIME_COL).sort_index()

    print("Data loaded. Shape:", df.shape)
    print("Columns:", df.columns.tolist())
    return df


# -----------------------------
# Step 2: Resample to daily and clean missing values
# -----------------------------

def resample_and_clean(df: pd.DataFrame, freq: str = RESAMPLE_FREQ) -> pd.DataFrame:
    """
    Resample the time series to the desired frequency (e.g., daily)
    and handle missing values.
    """
    print(f"\nResampling data to {freq} frequency using mean aggregation...")
    daily_df = df.resample(freq).mean()

    print("After resampling. Shape:", daily_df.shape)
    print("Missing values before filling:")
    print(daily_df.isna().sum())

    # Interpolate missing values over time for the target column
    daily_df[TARGET_COL] = daily_df[TARGET_COL].interpolate(method="time")

    # If still any missing, forward fill as backup
    daily_df[TARGET_COL] = daily_df[TARGET_COL].ffill()

    print("Missing values after filling:")
    print(daily_df.isna().sum())

    return daily_df


# -----------------------------
# Step 3: Select 12-month window (10 months train + 2 months test)
# -----------------------------

def select_one_year_window(df: pd.DataFrame,
                           start_date: str = YEAR_START,
                           end_date: str = YEAR_END) -> pd.DataFrame:
    """
    Select a continuous 12-month window from the full time series.
    """
    print(f"\nSelecting data from {start_date} to {end_date}...")
    mask = (df.index >= start_date) & (df.index <= end_date)
    year_df = df.loc[mask].copy()

    print("Selected window shape:", year_df.shape)
    print("Date range:", year_df.index.min(), "to", year_df.index.max())
    return year_df


def train_test_split_by_month(df: pd.DataFrame,
                              train_months: int = TRAIN_MONTHS):
    """
    Split the 12-month data into train (first train_months) and test (remaining).
    """
    if train_months >= 12:
        raise ValueError("train_months must be < 12 for a 10+2 split.")

    print(f"\nSplitting data into {train_months} months train and {12 - train_months} months test...")

    # Get the first timestamp and compute split date
    start = df.index.min()
    # Add train_months months to start
    split_date = start + pd.DateOffset(months=train_months)

    print("Train start:", start)
    print("Train end (split date):", split_date)

    train = df[df.index < split_date].copy()
    test = df[df.index >= split_date].copy()

    print("Train shape:", train.shape)
    print("Test shape:", test.shape)

    return train, test


# -----------------------------
# Step 4: Save processed data
# -----------------------------

def save_processed_data(full_df: pd.DataFrame,
                        train_df: pd.DataFrame,
                        test_df: pd.DataFrame):
    """
    Save cleaned full year, train, and test datasets as CSV.
    """
    full_path = os.path.join(OUTPUT_DATA_DIR, "cleaned_data_12months.csv")
    train_path = os.path.join(OUTPUT_DATA_DIR, "train_data_10months.csv")
    test_path = os.path.join(OUTPUT_DATA_DIR, "test_data_2months.csv")

    full_df.to_csv(full_path, index=True)
    train_df.to_csv(train_path, index=True)
    test_df.to_csv(test_path, index=True)

    print(f"\nSaved cleaned full data to: {full_path}")
    print(f"Saved train data to: {train_path}")
    print(f"Saved test data to: {test_path}")


# -----------------------------
# Step 5: Generate and save basic plots
# -----------------------------

def plot_time_series(df: pd.DataFrame, title: str, file_name: str):
    """
    Line plot of the time series.
    """
    plt.figure(figsize=(12, 5))
    plt.plot(df.index, df[TARGET_COL], label=TARGET_COL, color="tab:blue")
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel(TARGET_COL)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    save_path = os.path.join(PLOTS_DIR, file_name)
    plt.savefig(save_path)
    plt.close()
    print(f"Saved plot: {save_path}")


def plot_histogram(df: pd.DataFrame, title: str, file_name: str):
    """
    Histogram of the target variable.
    """
    plt.figure(figsize=(8, 5))
    sns.histplot(df[TARGET_COL], bins=30, kde=True, color="tab:green")
    plt.title(title)
    plt.xlabel(TARGET_COL)
    plt.ylabel("Frequency")
    plt.tight_layout()

    save_path = os.path.join(PLOTS_DIR, file_name)
    plt.savefig(save_path)
    plt.close()
    print(f"Saved plot: {save_path}")


# -----------------------------
# Main pipeline for Module 1
# -----------------------------

def run_module1():
    """
    Full Module 1 pipeline:
    - Load raw data
    - Resample & clean
    - Select 12-month window
    - Split into 10 months train + 2 months test
    - Save CSVs
    - Save basic plots
    """
    ensure_directories()

    # 1. Load
    input_path = os.path.join(DATA_DIR, INPUT_FILE)
    raw_df = load_data(input_path)

    # 2. Resample & clean
    daily_df = resample_and_clean(raw_df, RESAMPLE_FREQ)

    # 3. Select 12-month window
    year_df = select_one_year_window(daily_df, YEAR_START, YEAR_END)

    # 4. Split into train/test
    train_df, test_df = train_test_split_by_month(year_df, TRAIN_MONTHS)

    # 5. Save processed data
    save_processed_data(year_df, train_df, test_df)

    # 6. Generate and save plots
    plot_time_series(year_df,
                     title="Daily Energy Consumption - 12 Months",
                     file_name="timeseries_12months.png")

    plot_time_series(train_df,
                     title="Training Data - First 10 Months",
                     file_name="timeseries_train_10months.png")

    plot_time_series(test_df,
                     title="Test Data - Last 2 Months",
                     file_name="timeseries_test_2months.png")

    plot_histogram(year_df,
                   title="Distribution of Daily Energy Consumption (12 Months)",
                   file_name="histogram_12months.png")

    print("\nModule 1 preprocessing completed successfully.")


if __name__ == "__main__":
    run_module1()
