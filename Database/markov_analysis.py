
# MARKOV CHAIN ATTRIBUTION


import logging

import pandas as pd
from sqlalchemy import text

from db_connection import engine

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

EXCLUDED_STATES = {"START", "CONVERSION", "NULL"}


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
    """Parse timestamps and sort each user's journey chronologically."""
    logging.info("Preprocessing Data...")

    df["event_timestamp"] = pd.to_datetime(df["event_timestamp"])
    df = df.sort_values(["user_id", "session_id", "event_timestamp"])

    return df


def validate_data(df):
    """Log basic volume checks."""
    logging.info("Running Validation...")

    logging.info(f"Users     : {df['user_id'].nunique():,}")
    logging.info(f"Sessions  : {df['session_id'].nunique():,}")
    logging.info(f"Campaigns : {df['campaign_name'].nunique()}")
    logging.info(f"Platforms : {df['platform'].nunique()}")

    return df


def build_customer_paths(df):
    """Build a START -> ... -> CONVERSION/NULL event path for every session."""
    logging.info("Building Customer Paths...")

    customer_paths = (
        df.groupby(["user_id", "session_id"])["event_type"].apply(list).reset_index(name="Path")
    )

    customer_paths["Path"] = customer_paths["Path"].apply(
        lambda path: ["START"] + path + (["CONVERSION"] if "purchase" in path else ["NULL"])
    )

    logging.info(f"{len(customer_paths):,} customer paths created.")
    return customer_paths


def create_transition_matrix(customer_paths):
    """Count every state-to-state transition across all paths."""
    logging.info("Creating Transition Matrix...")

    transitions = [
        pair for path in customer_paths["Path"] for pair in zip(path[:-1], path[1:])
    ]
    transitions = pd.DataFrame(transitions, columns=["From_State", "To_State"])

    transition_matrix = transitions.groupby(["From_State", "To_State"]).size().reset_index(name="Count")

    logging.info("Transition Matrix Created.")
    return transition_matrix


def calculate_transition_probabilities(transition_matrix):
    """Normalize counts into per-state transition probabilities."""
    logging.info("Calculating Transition Probabilities...")

    transition_matrix["Probability"] = (
        transition_matrix["Count"] / transition_matrix.groupby("From_State")["Count"].transform("sum")
    ).round(4)

    logging.info("Transition Probabilities Calculated.")
    return transition_matrix


def calculate_removal_effect(customer_paths):
    """For each non-terminal state, measure the % drop in conversions if that state is removed."""
    logging.info("Calculating Removal Effect...")

    paths = customer_paths["Path"]
    original_conversions = paths.apply(lambda p: "CONVERSION" in p).sum()

    all_states = {s for path in paths for s in path}
    removable_states = sorted(all_states - EXCLUDED_STATES)

    results = []
    for state in removable_states:
        modified_paths = paths.apply(lambda path: [s for s in path if s != state])
        remaining_conversions = modified_paths.apply(lambda p: "CONVERSION" in p).sum()

        removal_effect = ((original_conversions - remaining_conversions) / original_conversions) * 100

        results.append(
            {
                "State": state,
                "Original_Conversions": original_conversions,
                "Remaining_Conversions": remaining_conversions,
                "Removal_Effect(%)": round(removal_effect, 2),
            }
        )

    removal_effect_df = pd.DataFrame(results).sort_values("Removal_Effect(%)", ascending=False)

    logging.info("Removal Effect Calculated.")
    return removal_effect_df


def save_results(tables):
    """Persist each attribution table to MySQL and CSV."""
    logging.info("Saving Attribution Tables...")

    for table_name, df in tables.items():
        df.to_sql(table_name, engine, if_exists="replace", index=False)
        df.to_csv(f"{table_name}.csv", index=False)
        logging.info(f"{table_name} saved successfully.")


def display_summary(transition_probabilities, removal_effect):
    """Print transition probabilities and the removal-effect ranking."""
    print("\n" + "=" * 90)
    print("TRANSITION PROBABILITIES")
    print("=" * 90)
    print(transition_probabilities)

    print("\n" + "=" * 90)
    print("REMOVAL EFFECT (Channel Importance)")
    print("=" * 90)
    print(removal_effect)

    print(f"\nTop Channel by Removal Effect : {removal_effect.iloc[0]['State']}")
    print(f"Removal Effect                : {removal_effect.iloc[0]['Removal_Effect(%)']:.2f}%")


def main():
    try:
        df = load_data()
        df = preprocess_data(df)
        validate_data(df)

        customer_paths = build_customer_paths(df)

        transition_matrix = create_transition_matrix(customer_paths)
        transition_probabilities = calculate_transition_probabilities(transition_matrix)
        removal_effect = calculate_removal_effect(customer_paths)

        save_results(
            {
                "markov_transition_probabilities": transition_probabilities,
                "markov_removal_effect": removal_effect,
            }
        )

        display_summary(transition_probabilities, removal_effect)

        logging.info("Markov Chain Attribution Completed Successfully.")

    except Exception as e:
        logging.exception(e)


if __name__ == "__main__":
    main()