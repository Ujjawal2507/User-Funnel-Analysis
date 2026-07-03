
# COHORT ANALYSIS


import logging

import numpy as np
import pandas as pd
from sqlalchemy import text

from db_connection import engine

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


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
    """Parse timestamps and add date/month features used for cohorting."""
    logging.info("Preprocessing data...")

    df["event_timestamp"] = pd.to_datetime(df["event_timestamp"])
    df["date"] = df["event_timestamp"].dt.date
    df["year"] = df["event_timestamp"].dt.year
    df["month"] = df["event_timestamp"].dt.month
    df["month_name"] = df["event_timestamp"].dt.month_name()
    df["year_month"] = df["event_timestamp"].dt.to_period("M")

    return df


def validate_data(df):
    """Log basic volume and date-range checks."""
    logging.info("Running Validation...")

    logging.info(f"Users      : {df['user_id'].nunique():,}")
    logging.info(f"Sessions   : {df['session_id'].nunique():,}")
    logging.info(f"Revenue    : ₹{df['revenue'].sum():,.2f}")
    logging.info(f"Date Range : {df['date'].min()}  →  {df['date'].max()}")

    return df


def create_cohorts(df):
    """Assign each user a cohort month (first activity) and a cohort index (months since)."""
    logging.info("Creating Customer Cohorts...")

    first_activity = df.groupby("user_id")["year_month"].min().rename("cohort_month")
    df = df.merge(first_activity, on="user_id", how="left")

    df["cohort_index"] = (df["year_month"].dt.year - df["cohort_month"].dt.year) * 12 + (
        df["year_month"].dt.month - df["cohort_month"].dt.month
    )

    logging.info("Cohorts Created Successfully.")
    return df


def calculate_retention(df):
    """Build a cohort_month x cohort_index retention matrix (% of cohort still active)."""
    logging.info("Calculating Cohort Retention...")

    cohort_data = (
        df.groupby(["cohort_month", "cohort_index"])["user_id"].nunique().reset_index(name="Users")
    )

    cohort_size = cohort_data.query("cohort_index == 0")[["cohort_month", "Users"]].rename(
        columns={"Users": "Cohort_Size"}
    )

    cohort_data = cohort_data.merge(cohort_size, on="cohort_month", how="left")
    cohort_data["Retention(%)"] = (cohort_data["Users"] / cohort_data["Cohort_Size"]) * 100

    retention_matrix = cohort_data.pivot(
        index="cohort_month", columns="cohort_index", values="Retention(%)"
    ).round(2)

    logging.info("Retention Matrix Created.")
    return retention_matrix, cohort_data


def calculate_revenue_retention(df):
    """Build a cohort_month x cohort_index matrix of total revenue."""
    logging.info("Calculating Revenue Retention...")

    revenue_retention = (
        df.groupby(["cohort_month", "cohort_index"])
        .agg(Revenue=("revenue", "sum"))
        .reset_index()
        .pivot(index="cohort_month", columns="cohort_index", values="Revenue")
        .fillna(0)
        .round(2)
    )

    return revenue_retention


def calculate_purchase_retention(df):
    """Build a cohort_month x cohort_index matrix of purchase counts."""
    logging.info("Calculating Purchase Retention...")

    purchase_retention = (
        df[df["event_type"] == "purchase"]
        .groupby(["cohort_month", "cohort_index"])
        .agg(Purchases=("order_id", "count"))
        .reset_index()
        .pivot(index="cohort_month", columns="cohort_index", values="Purchases")
        .fillna(0)
    )

    return purchase_retention


def calculate_ltv(df):
    """Compute average revenue per user for each cohort month."""
    logging.info("Calculating Customer Lifetime Value...")

    ltv = df.groupby("cohort_month").agg(Users=("user_id", "nunique"), Revenue=("revenue", "sum")).reset_index()
    ltv["LTV"] = np.where(ltv["Users"] == 0, 0, ltv["Revenue"] / ltv["Users"])

    return ltv.round(2)


def save_results(tables):
    """Persist each cohort table to MySQL and CSV. Period columns are stringified for compatibility."""
    logging.info("Saving Cohort Tables...")

    for table_name, df in tables.items():
        out = df.reset_index() if df.index.name else df.copy()

        for col in out.columns:
            if pd.api.types.is_period_dtype(out[col]):
                out[col] = out[col].astype(str)

        out.to_sql(table_name, engine, if_exists="replace", index=False)
        out.to_csv(f"{table_name}.csv", index=False)
        logging.info(f"{table_name} saved successfully.")


def display_summary(retention_matrix, revenue_retention, purchase_retention, ltv):
    """Print the key cohort tables and headline LTV stats."""
    print("\n" + "=" * 90)
    print("COHORT RETENTION (%)")
    print("=" * 90)
    print(retention_matrix)

    print("\n" + "=" * 90)
    print("REVENUE RETENTION (₹)")
    print("=" * 90)
    print(revenue_retention)

    print("\n" + "=" * 90)
    print("PURCHASE RETENTION (count)")
    print("=" * 90)
    print(purchase_retention)

    print("\n" + "=" * 90)
    print("CUSTOMER LIFETIME VALUE")
    print("=" * 90)
    print(ltv)

    print(f"\nAverage LTV across cohorts : ₹{ltv['LTV'].mean():,.2f}")
    print(f"Best Cohort                : {ltv.loc[ltv['LTV'].idxmax(), 'cohort_month']}")


def main():
    try:
        df = load_data()
        df = preprocess_data(df)
        validate_data(df)
        df = create_cohorts(df)

        retention_matrix, cohort_data = calculate_retention(df)
        revenue_retention = calculate_revenue_retention(df)
        purchase_retention = calculate_purchase_retention(df)
        ltv = calculate_ltv(df)

        save_results(
            {
                "cohort_retention": retention_matrix,
                "cohort_revenue_retention": revenue_retention,
                "cohort_purchase_retention": purchase_retention,
                "cohort_ltv": ltv,
            }
        )

        display_summary(retention_matrix, revenue_retention, purchase_retention, ltv)

        logging.info("Cohort Analysis Completed Successfully.")

    except Exception as e:
        logging.exception(e)


if __name__ == "__main__":
    main()