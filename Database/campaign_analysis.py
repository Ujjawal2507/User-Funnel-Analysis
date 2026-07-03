

import numpy as np
import pandas as pd
from sqlalchemy import text

from db_connection import engine

EVENT_RENAME = {
    "impression": "impressions",
    "click": "clicks",
    "landing_page": "landing_pages",
    "product_view": "product_views",
    "add_to_cart": "add_to_cart",
    "purchase": "purchases",
}

REQUIRED_EVENT_COLS = list(EVENT_RENAME.values())

FUNNEL_STEPS = {
    "CTR": ("clicks", "impressions"),
    "Landing_Rate": ("landing_pages", "clicks"),
    "Product_View_Rate": ("product_views", "landing_pages"),
    "Cart_Rate": ("add_to_cart", "product_views"),
    "Purchase_Rate": ("purchases", "add_to_cart"),
    "Overall_Conversion": ("purchases", "impressions"),
}


def load_data():
    """Load raw user journey events from MySQL."""
    print("=" * 70)
    print("Loading User Journey Data...")
    print("=" * 70)

    df = pd.read_sql(text("SELECT * FROM user_journey_events"), engine)

    print(f"Loaded {len(df):,} rows, {df.shape[1]} columns.\n")
    return df


def explore_data(df):
    """Print quick EDA: dtypes, missing values, duplicates, basic counts."""
    print("=" * 70)
    print("DATASET OVERVIEW")
    print("=" * 70)
    print(df.dtypes, "\n")
    print("Missing values:\n", df.isnull().sum(), "\n")
    print(f"Duplicate rows: {df.duplicated().sum()}\n")

    print(f"Unique Users    : {df['user_id'].nunique():,}")
    print(f"Unique Sessions : {df['session_id'].nunique():,}")
    print(f"Companies       : {df['company_name'].nunique()}")
    print(f"Campaigns       : {df['campaign_name'].nunique()}")
    print(f"Platforms       : {df['platform'].nunique()}")
    print(f"Cities          : {df['city'].nunique()}")
    print(f"Industries      : {df['industry'].nunique()}\n")

    print("Event distribution:\n", df["event_type"].value_counts().sort_index(), "\n")
    print("Platform distribution:\n", df["platform"].value_counts(), "\n")
    print("Top campaigns:\n", df["campaign_name"].value_counts(), "\n")


def preprocess_data(df):
    """Parse timestamps and add date/time features."""
    df["event_timestamp"] = pd.to_datetime(df["event_timestamp"])
    df["date"] = df["event_timestamp"].dt.date
    df["year"] = df["event_timestamp"].dt.year
    df["month"] = df["event_timestamp"].dt.month
    df["month_name"] = df["event_timestamp"].dt.month_name()
    df["day"] = df["event_timestamp"].dt.day
    df["weekday"] = df["event_timestamp"].dt.day_name()
    df["hour"] = df["event_timestamp"].dt.hour
    return df


def calculate_campaign_metrics(df):
    """Aggregate event counts, revenue, users, and sessions per campaign."""
    event_counts = (
        df.groupby(["campaign_name", "event_type"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
        .rename(columns=EVENT_RENAME)
    )

    for col in REQUIRED_EVENT_COLS:
        if col not in event_counts.columns:
            event_counts[col] = 0

    revenue = df.groupby("campaign_name")["revenue"].sum().reset_index(name="total_revenue")
    users = df.groupby("campaign_name")["user_id"].nunique().reset_index(name="unique_users")
    sessions = df.groupby("campaign_name")["session_id"].nunique().reset_index(name="total_sessions")

    summary = (
        event_counts
        .merge(revenue, on="campaign_name")
        .merge(users, on="campaign_name")
        .merge(sessions, on="campaign_name")
    )
    return summary


def calculate_campaign_kpis(summary):
    """Compute funnel conversion rates, AOV, and cart abandonment."""
    for metric, (num, den) in FUNNEL_STEPS.items():
        summary[metric] = np.where(summary[den] == 0, 0, (summary[num] / summary[den]) * 100)

    summary["Average_Order_Value"] = np.where(
        summary["purchases"] == 0, 0, summary["total_revenue"] / summary["purchases"]
    )

    summary["Cart_Abandonment"] = np.where(
        summary["add_to_cart"] == 0,
        0,
        (1 - summary["purchases"] / summary["add_to_cart"]) * 100,
    )

    summary.replace([np.inf, -np.inf], 0, inplace=True)
    summary.fillna(0, inplace=True)

    numeric_cols = summary.select_dtypes(include=np.number).columns
    summary[numeric_cols] = summary[numeric_cols].round(2)

    return summary


def save_results(summary):
    """Persist campaign summary to MySQL and CSV."""
    summary.to_sql("campaign_summary", engine, if_exists="replace", index=False)
    summary.to_csv("campaign_summary.csv", index=False)
    print("\nCampaign Summary saved successfully.")


def display_summary(summary):
    """Print a readable summary of campaign performance."""
    print("\n" + "=" * 70)
    print("CAMPAIGN ANALYSIS SUMMARY")
    print("=" * 70)

    print(f"Total Campaigns    : {len(summary)}")
    print(f"Total Revenue      : ₹{summary['total_revenue'].sum():,.2f}")
    print(f"Average CTR        : {summary['CTR'].mean():.2f}%")
    print(f"Average Conversion : {summary['Overall_Conversion'].mean():.2f}%")
    print(f"Average AOV        : ₹{summary['Average_Order_Value'].mean():,.2f}")

    print("\nTop 5 Campaigns by Revenue\n")
    print(
        summary[["campaign_name", "total_revenue", "CTR", "Overall_Conversion"]]
        .sort_values("total_revenue", ascending=False)
        .head()
    )


def main():
    df = load_data()
    explore_data(df)
    df = preprocess_data(df)

    summary = calculate_campaign_metrics(df)
    summary = calculate_campaign_kpis(summary)

    summary.sort_values("total_revenue", ascending=False, inplace=True)
    summary.reset_index(drop=True, inplace=True)

    save_results(summary)
    display_summary(summary)


if __name__ == "__main__":
    main()