# ==========================================================
# A/B TESTING
# ==========================================================

import logging

import numpy as np
import pandas as pd
from scipy.stats import norm
from sqlalchemy import text

from db_connection import engine

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

VARIANT_COL = "ad_variant"

REQUIRED_EVENT_COLS = ["impression", "click", "landing_page", "product_view", "add_to_cart", "purchase"]

FUNNEL_METRICS = {
    "CTR": ("click", "impression"),
    "Landing_Rate": ("landing_page", "click"),
    "Product_View_Rate": ("product_view", "landing_page"),
    "Cart_Rate": ("add_to_cart", "product_view"),
    "Purchase_Rate": ("purchase", "add_to_cart"),
    "Overall_Conversion": ("purchase", "impression"),
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
    """Parse event timestamps."""
    logging.info("Preprocessing data...")
    df["event_timestamp"] = pd.to_datetime(df["event_timestamp"])
    return df


def prepare_ab_summary(df):
    """Aggregate event counts and revenue per campaign/variant combination."""
    logging.info("Preparing Variant Summary...")

    events = (
        df.groupby(["campaign_name", VARIANT_COL, "event_type"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )

    revenue = (
        df.groupby(["campaign_name", VARIANT_COL])["revenue"].sum().reset_index(name="Revenue")
    )

    summary = events.merge(revenue, on=["campaign_name", VARIANT_COL])

    logging.info("Variant Summary Created.")
    return summary


def calculate_metrics(summary):
    """Compute funnel conversion rates and revenue-efficiency metrics per variant."""
    logging.info("Calculating A/B Test Metrics...")

    for col in REQUIRED_EVENT_COLS:
        if col not in summary.columns:
            summary[col] = 0

    for metric, (num, den) in FUNNEL_METRICS.items():
        summary[metric] = np.where(summary[den] == 0, 0, (summary[num] / summary[den]) * 100)

    summary["Revenue_Per_Click"] = np.where(
        summary["click"] == 0, 0, summary["Revenue"] / summary["click"]
    )

    summary["Revenue_Per_Impression"] = np.where(
        summary["impression"] == 0, 0, summary["Revenue"] / summary["impression"]
    )

    summary["Average_Order_Value"] = np.where(
        summary["purchase"] == 0, 0, summary["Revenue"] / summary["purchase"]
    )

    numeric_cols = summary.select_dtypes(include=np.number).columns
    summary[numeric_cols] = summary[numeric_cols].round(2)

    logging.info("Metric Calculation Completed.")
    return summary


def run_z_test(summary, alpha=0.05):
    """Run a two-proportion z-test (impressions -> purchases) between variants within each campaign."""
    logging.info("Running A/B Statistical Tests...")

    results = []

    for campaign, group in summary.groupby("campaign_name"):
        if len(group) != 2:
            continue

        group = group.sort_values(VARIANT_COL).reset_index(drop=True)
        A, B = group.iloc[0], group.iloc[1]

        n1, n2 = A["impression"], B["impression"]
        x1, x2 = A["purchase"], B["purchase"]

        if n1 == 0 or n2 == 0:
            continue

        p1, p2 = x1 / n1, x2 / n2
        pooled = (x1 + x2) / (n1 + n2)
        standard_error = np.sqrt(pooled * (1 - pooled) * ((1 / n1) + (1 / n2)))

        if standard_error == 0:
            continue

        z_score = (p2 - p1) / standard_error
        p_value = 2 * (1 - norm.cdf(abs(z_score)))
        significant = p_value < alpha

        if p2 > p1:
            winner = B[VARIANT_COL]
        elif p1 > p2:
            winner = A[VARIANT_COL]
        else:
            winner = "Tie"

        results.append(
            {
                "campaign_name": campaign,
                "Variant_A": A[VARIANT_COL],
                "Variant_B": B[VARIANT_COL],
                "Conversion_A": round(p1 * 100, 2),
                "Conversion_B": round(p2 * 100, 2),
                "Z_Score": round(z_score, 3),
                "P_Value": round(p_value, 5),
                "Significant": significant,
                "Winner": winner,
            }
        )

    logging.info("Z-Test Completed.")
    return pd.DataFrame(results)


def save_results(summary, results):
    """Persist the variant summary, full test results, and winners table."""
    logging.info("Saving A/B Test Results...")

    tables = {
        "ab_test_summary": summary,
        "ab_test_results": results,
        "campaign_winners": results[["campaign_name", "Winner", "Significant", "P_Value"]],
    }

    for table_name, dataframe in tables.items():
        dataframe.to_sql(table_name, engine, if_exists="replace", index=False)
        dataframe.to_csv(f"{table_name}.csv", index=False)
        logging.info(f"{table_name} saved successfully.")


def display_summary(results):
    """Print full test results and a quick significance/winner breakdown."""
    print("\n" + "=" * 100)
    print("A/B TEST RESULTS")
    print("=" * 100)
    print(results)

    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)

    total_campaigns = len(results)
    significant_tests = results["Significant"].sum()

    print(f"Total Campaigns Tested      : {total_campaigns}")
    print(f"Statistically Significant   : {significant_tests}")
    print(f"Not Significant             : {total_campaigns - significant_tests}")

    print("\nWinner Distribution")
    print(results["Winner"].value_counts())


def main():
    try:
        df = load_data()
        df = preprocess_data(df)

        summary = prepare_ab_summary(df)
        summary = calculate_metrics(summary)

        results = run_z_test(summary)

        save_results(summary, results)
        display_summary(results)

        logging.info("A/B Testing Completed Successfully.")

    except Exception as e:
        logging.exception(e)


if __name__ == "__main__":
    main()