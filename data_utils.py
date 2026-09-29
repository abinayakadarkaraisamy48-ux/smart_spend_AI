from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st

DATA_PATH = Path(__file__).resolve().parent"SmartSpendAI_cleaned_dataset.csv"

EXPECTED_COLUMNS = [
    "transaction_id", "payer_name", "payee_name", "date", "time",
    "amount", "category", "payment_mode", "city", "transaction_status"
]

@st.cache_data
def load_dataset():
    return pd.read_csv(DATA_PATH)

def validate_dataset(df):
    return [col for col in EXPECTED_COLUMNS if col not in df.columns]

def preprocess_data(df):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df["time"] = df["time"].astype(str)
    df = df.dropna(subset=["date", "amount", "category"])
    return df

def create_features(df):
    df = df.copy()
    df["month"] = df["date"].dt.month
    df["month_name"] = df["date"].dt.month_name()
    df["day"] = df["date"].dt.day
    df["day_of_week"] = df["date"].dt.dayofweek
    df["day_name"] = df["date"].dt.day_name()
    df["hour"] = pd.to_datetime(df["time"], format="%H:%M", errors="coerce").dt.hour.fillna(12)
    df["weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["spending_level"] = pd.cut(
        df["amount"],
        bins=[-np.inf, 200, 500, 1000, 3000, np.inf],
        labels=["Very Low", "Low", "Medium", "High", "Very High"]
    )
    return df
