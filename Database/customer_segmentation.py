
# CUSTOMER SEGMENTATION


import logging

import numpy as np
import pandas as pd
from sqlalchemy import text

from db_connection import engine

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

REQUIRED_EVENT_COLS = ["impression", "click", "landing_page", "product_view", "add_to_cart", "purchase"]

FUNNEL_KPIS = {
    "CTR": ("click", "impression"),
    "Landing_Rate": ("landing_page", "click"),
    "Product_View_Rate": ("product_view", "landing_page"),
    "Cart_Rate": ("add_to_cart", "product_view"),
    "Purchase_Rate": ("purchase", "add_to_cart"),
    "Overall_Conversion": ("purchase", "impression"),
}

SEGMENT_COLUMNS = {
    "age_summary": "age_group",
    "city_summary": "city",
    "device_summary": "device_type",
    "traffic_summary": "traffic_source",
    "industry_summary": "industry",
    "lifecycle_summary": "user_lifecycle",
}


def load_data():
    """Load raw user journey events from MySQL."""
    try:
        logging.info("Loading User Journey Data...")
        df = pd.read_sql(text("SELECT * FROM user_journey_events"), engine)
        logging.info(f"{len(df):,} records loaded.")
        return df
    except Exception as e:
        logging.exception(e)
        raise


def preprocess_data(df):
    """Parse timestamps and add date/time features."""
    logging.info("Preprocessing data...")

    df["event_timestamp"] = pd.to_datetime(df["event_timestamp"])
    df["date"] = df["event_timestamp"].dt.date
    df["month"] = df["event_timestamp"].dt.month_name()
    df["weekday"] = df["event_timestamp"].dt.day_name()
    df["hour"] = df["event_timestamp"].dt.hour

    return df


def validate_data(df):
    """Log basic data quality and volume checks."""
    logging.info("Running validation...")

    logging.info(f"Rows      : {len(df):,}")
    logging.info(f"Users     : {df['user_id'].nunique():,}")
    logging.info(f"Sessions  : {df['session_id'].nunique():,}")
    logging.info(f"Purchases : {(df['event_type'] == 'purchase').sum():,}")
    logging.info(f"Revenue   : ₹{df['revenue'].sum():,.2f}")

    return df


def build_segmentation_summary(df, segment_column):
    """Aggregate users, sessions, revenue, purchases, and event counts by segment."""
    logging.info(f"Building {segment_column} Summary...")

    summary = (
        df.groupby(segment_column)
        .agg(
            Users=("user_id", "nunique"),
            Sessions=("session_id", "nunique"),
            Revenue=("revenue", "sum"),
            Purchases=("order_id", "count"),
        )
        .reset_index()
    )

    events = df.pivot_table(
        index=segment_column, columns="event_type", values="user_id", aggfunc="count", fill_value=0
    ).reset_index()

    summary = summary.merge(events, on=segment_column, how="left")
    summary.fillna(0, inplace=True)

    logging.info(f"{segment_column} Summary Created.")
    return summary


def calculate_kpis(summary):
    """Compute funnel conversion rates, AOV, revenue ratios, and cart abandonment."""
    logging.info("Calculating KPIs...")

    for col in REQUIRED_EVENT_COLS:
        if col not in summary.columns:
            summary[col] = 0

    for metric, (num, den) in FUNNEL_KPIS.items():
        summary[metric] = np.where(summary[den] == 0, 0, (summary[num] / summary[den]) * 100)

    summary["Average_Order_Value"] = np.where(
        summary["Purchases"] == 0, 0, summary["Revenue"] / summary["Purchases"]
    )

    summary["Revenue_Per_User"] = np.where(summary["Users"] == 0, 0, summary["Revenue"] / summary["Users"])

    summary["Revenue_Per_Session"] = np.where(
        summary["Sessions"] == 0, 0, summary["Revenue"] / summary["Sessions"]
    )

    summary["Cart_Abandonment"] = np.where(
        summary["add_to_cart"] == 0,
        0,
        (1 - summary["purchase"] / summary["add_to_cart"]) * 100,
    )

    numeric_cols = summary.select_dtypes(include=np.number).columns
    summary[numeric_cols] = summary[numeric_cols].round(2)

    logging.info("KPIs Calculated Successfully.")
    return summary


def save_results(summary_tables):
    """Persist each summary table to MySQL and CSV."""
    logging.info("Saving Summary Tables...")

    for table_name, df in summary_tables.items():
        df.to_sql(table_name, engine, if_exists="replace", index=False)
        df.to_csv(f"{table_name}.csv", index=False)
        logging.info(f"{table_name} saved successfully.")


def display_summary(summary_tables):
    """Print a readable overview of every segmentation table."""
    print("\n" + "=" * 80)
    print("CUSTOMER SEGMENTATION SUMMARY")
    print("=" * 80)

    for table_name, df in summary_tables.items():
        print(f"\n{table_name.upper()}")
        print("-" * 80)
        print(df.head())
        print(f"\nTotal Segments      : {len(df)}")
        print(f"Total Revenue        : ₹{df['Revenue'].sum():,.2f}")
        print(f"Average Conversion   : {df['Overall_Conversion'].mean():.2f}%")
        print("-" * 80)


def main():
    try:
        df = load_data()
        df = preprocess_data(df)
        validate_data(df)

        summary_tables = {
            table_name: calculate_kpis(build_segmentation_summary(df, segment_column))
            for table_name, segment_column in SEGMENT_COLUMNS.items()
        }

        save_results(summary_tables)
        display_summary(summary_tables)

        logging.info("Customer Segmentation Completed Successfully.")

    except Exception as e:
        logging.exception(e)


if __name__ == "__main__":
    main()