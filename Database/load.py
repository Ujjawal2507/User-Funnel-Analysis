import pandas as pd
from db_connection import engine

df = pd.read_csv(r"C:\Users\Ujjawal Asthana\OneDrive\Desktop\User Funnel Analysis\Synthetic_Data\user_journey_events.csv")

df.to_sql(
    "user_journey_events",
    engine,
    if_exists="append",
    index=False
)

print("Data loaded successfully!")
