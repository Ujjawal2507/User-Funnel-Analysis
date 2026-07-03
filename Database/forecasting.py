
# FORECASTING USING ARIMA


import logging
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sqlalchemy import text
from statsmodels.tsa.arima.model import ARIMA

from db_connection import engine

warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

METRICS = ["Revenue", "Purchases", "Sessions", "Conversion_Rate"]


def load_data():
    """Load raw user journey events from MySQL."""
    try:
        logging.info("Loading User Journey Data...")
        df = pd.read_sql(text("SELECT * FROM user_journey_events"), engine)
        logging.info(f"{len(df):,} rows loaded.")
        return df
    except Exception as e:
        logging.exception(e)
        raise


def preprocess_data(df):
    """Parse timestamps and derive a monthly period column."""
    logging.info("Preprocessing Data...")
    df["event_timestamp"] = pd.to_datetime(df["event_timestamp"])
    df["year_month"] = df["event_timestamp"].dt.to_period("M").dt.to_timestamp()
    return df


def prepare_time_series(df):
    """Aggregate revenue, purchases, sessions, and conversion rate by month."""
    logging.info("Preparing Monthly Time Series...")

    monthly_data = (
        df.groupby("year_month")
        .agg(
            Revenue=("revenue", "sum"),
            Purchases=("order_id", "count"),
            Sessions=("session_id", "nunique"),
            Impressions=("event_type", lambda x: (x == "impression").sum()),
        )
        .reset_index()
    )

    monthly_data["Conversion_Rate"] = np.where(
        monthly_data["Impressions"] == 0,
        0,
        (monthly_data["Purchases"] / monthly_data["Impressions"]) * 100,
    )

    logging.info("Monthly Time Series Created.")
    return monthly_data


def forecast_metric(monthly_data, metric, order=(1, 1, 1), forecast_periods=3):
    """Fit an ARIMA model on a monthly metric and forecast N future periods."""
    logging.info(f"Forecasting {metric}...")

    ts = monthly_data.set_index("year_month")[metric].astype(float)

    fitted_model = ARIMA(ts, order=order).fit()
    forecast = fitted_model.forecast(steps=forecast_periods)

    future_dates = pd.date_range(
        start=ts.index[-1] + pd.DateOffset(months=1), periods=forecast_periods, freq="MS"
    )

    forecast_df = pd.DataFrame({"year_month": future_dates, metric: forecast.values, "Type": "Forecast"})
    historical_df = pd.DataFrame({"year_month": ts.index, metric: ts.values, "Type": "Historical"})

    result = pd.concat([historical_df, forecast_df], ignore_index=True)
    result[metric] = result[metric].round(2)

    logging.info(f"{metric} Forecast Completed.")
    return result, fitted_model


def evaluate_forecast(model, monthly_data, metric):
    """Compute in-sample fit quality (MAE, RMSE, AIC, BIC) for a fitted model."""
    logging.info(f"Evaluating {metric} Forecast...")

    actual = monthly_data[metric].astype(float).values[1:]
    fitted = model.fittedvalues.values[1:]

    return {
        "Metric": metric,
        "MAE": round(mean_absolute_error(actual, fitted), 2),
        "RMSE": round(np.sqrt(mean_squared_error(actual, fitted)), 2),
        "AIC": round(model.aic, 2),
        "BIC": round(model.bic, 2),
    }


def save_results(forecasts, evaluation):
    """Persist each forecast table and the evaluation summary to MySQL and CSV."""
    logging.info("Saving Forecast Results...")

    for table_name, df in forecasts.items():
        df.to_sql(table_name, engine, if_exists="replace", index=False)
        df.to_csv(f"{table_name}.csv", index=False)

    evaluation_df = pd.DataFrame(evaluation)
    evaluation_df.to_sql("forecast_summary", engine, if_exists="replace", index=False)
    evaluation_df.to_csv("forecast_summary.csv", index=False)

    logging.info("Forecast Results Saved.")


def display_summary(evaluation):
    """Print the model evaluation table."""
    evaluation_df = pd.DataFrame(evaluation)

    print("\n" + "=" * 90)
    print("FORECAST SUMMARY")
    print("=" * 90)
    print(evaluation_df)

    print("\n" + "=" * 90)
    print("MODEL QUALITY")
    print("=" * 90)
    print(evaluation_df[["Metric", "MAE", "RMSE", "AIC"]])


def main():
    try:
        df = load_data()
        df = preprocess_data(df)
        monthly_data = prepare_time_series(df)

        forecasts = {}
        evaluation = []

        for metric in METRICS:
            forecast_df, model = forecast_metric(monthly_data, metric)
            forecasts[f"forecast_{metric.lower()}"] = forecast_df
            evaluation.append(evaluate_forecast(model, monthly_data, metric))

        save_results(forecasts, evaluation)
        display_summary(evaluation)

        logging.info("Forecasting Completed Successfully.")

    except Exception as e:
        logging.exception(e)


if __name__ == "__main__":
    main()