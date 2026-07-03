
# USER JOURNEY FUNNEL ANALYSIS


import logging

import numpy as np
import pandas as pd
from sqlalchemy import text

from db_connection import engine

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

FUNNEL_ORDER = ["impression", "click", "landing_page", "product_view", "add_to_cart", "purchase"]


def load_data():
    """Load raw user journey events from MySQL."""
    try:
        logging.info("Loading data from MySQL...")
        df = pd.read_sql(text("SELECT * FROM user_journey_events"), engine)
        logging.info(f"{len(df):,} rows loaded.")
        return df
    except Exception as e:
        logging.error(e)
        raise


def preprocess_data(df):
    """Parse timestamps and add date/time features."""
    logging.info("Preprocessing data...")

    df["event_timestamp"] = pd.to_datetime(df["event_timestamp"])
    df["date"] = df["event_timestamp"].dt.date
    df["month"] = df["event_timestamp"].dt.month_name()
    df["weekday"] = df["event_timestamp"].dt.day_name()
    df["hour"] = df["event_timestamp"].dt.hour

    logging.info("Preprocessing completed.")
    return df


def validate_data(df):
    """Log basic data quality checks."""
    logging.info("Running validation checks...")

    logging.info(f"Missing Values  : {df.isnull().sum().sum()}")
    logging.info(f"Duplicate Rows  : {df.duplicated().sum()}")
    logging.info(f"Unique Users    : {df['user_id'].nunique():,}")
    logging.info(f"Unique Sessions : {df['session_id'].nunique():,}")

    return df


def build_funnel(df):
    """Count unique users and sessions at each funnel stage, in order."""
    logging.info("Building Funnel Summary...")

    funnel_summary = (
        df.groupby("event_type")
        .agg(Users=("user_id", "nunique"), Sessions=("session_id", "nunique"))
        .reindex(FUNNEL_ORDER)
        .fillna(0)
        .reset_index()
        .rename(columns={"event_type": "Stage"})
    )

    logging.info("Funnel Summary Created.")
    return funnel_summary


def calculate_funnel_kpis(funnel_summary):
    """Compute stage/overall conversion, drop-off, and funnel efficiency."""
    logging.info("Calculating Funnel KPI's...")

    total_users = funnel_summary.loc[0, "Users"]

    funnel_summary["Previous_Stage_Users"] = funnel_summary["Users"].shift(1)
    funnel_summary.loc[0, "Previous_Stage_Users"] = total_users

    funnel_summary["Stage_Conversion(%)"] = np.where(
        funnel_summary["Previous_Stage_Users"] == 0,
        0,
        (funnel_summary["Users"] / funnel_summary["Previous_Stage_Users"]) * 100,
    )

    funnel_summary["Overall_Conversion(%)"] = (funnel_summary["Users"] / total_users) * 100

    funnel_summary["Dropoff_Users"] = (
        funnel_summary["Previous_Stage_Users"] - funnel_summary["Users"]
    ).clip(lower=0)

    funnel_summary["Dropoff(%)"] = np.where(
        funnel_summary["Previous_Stage_Users"] == 0,
        0,
        (funnel_summary["Dropoff_Users"] / funnel_summary["Previous_Stage_Users"]) * 100,
    )

    funnel_summary["Cumulative_Dropoff"] = total_users - funnel_summary["Users"]
    funnel_summary["Funnel_Efficiency"] = (funnel_summary["Users"] / total_users) * 100

    numeric_cols = funnel_summary.select_dtypes(include=np.number).columns
    funnel_summary[numeric_cols] = funnel_summary[numeric_cols].round(2)

    logging.info("Funnel KPI Calculation Completed.")
    return funnel_summary


def save_results(funnel_summary):
    """Persist funnel summary to MySQL and CSV."""
    logging.info("Saving Results...")

    funnel_summary.to_sql("funnel_summary", engine, if_exists="replace", index=False)
    funnel_summary.to_csv("funnel_summary.csv", index=False)

    logging.info("Results Saved Successfully.")


def display_summary(funnel_summary):
    """Print the funnel table and key headline stats."""
    logging.info("Displaying Funnel Summary...\n")

    print("=" * 90)
    print("FUNNEL ANALYSIS SUMMARY")
    print("=" * 90)
    print(funnel_summary)

    print("\n" + "=" * 90)
    print("OVERALL STATISTICS")
    print("=" * 90)

    top_dropoff_idx = funnel_summary["Dropoff(%)"].idxmax()

    print(f"Total Users             : {int(funnel_summary.iloc[0]['Users']):,}")
    print(f"Final Purchases         : {int(funnel_summary.iloc[-1]['Users']):,}")
    print(f"Overall Conversion      : {funnel_summary.iloc[-1]['Overall_Conversion(%)']:.2f}%")
    print(f"Highest Drop-off Stage  : {funnel_summary.loc[top_dropoff_idx, 'Stage']}")
    print(f"Highest Drop-off %      : {funnel_summary['Dropoff(%)'].max():.2f}%")


def main():
    try:
        df = load_data()
        df = preprocess_data(df)
        validate_data(df)

        funnel_summary = build_funnel(df)
        funnel_summary = calculate_funnel_kpis(funnel_summary)

        save_results(funnel_summary)
        display_summary(funnel_summary)

        logging.info("Funnel Analysis Completed Successfully.")

    except Exception as e:
        logging.exception(e)


if __name__ == "__main__":
    main()