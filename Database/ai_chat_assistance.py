# ==========================================================
# AI CHAT ASSISTANT
# ==========================================================
import os

import pandas as pd
import google.generativeai as genai
from dotenv import load_dotenv
from sqlalchemy import text
from db_connection import engine

load_dotenv()

# ==========================================================
# GEMINI CONFIGURATION
# Read from the environment instead of being hardcoded.
# ==========================================================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise RuntimeError("Set the GEMINI_API_KEY environment variable before running this script.")

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-2.5-flash")

# ==========================================================
# TABLE ROUTER
# ==========================================================
TABLE_MAPPING = {
    "campaign": "campaign_summary",
    "roas": "campaign_summary",
    "ctr": "campaign_summary",
    "funnel": "funnel_summary",
    "drop": "funnel_summary",
    "customer": "customer_summary",
    "segment": "customer_summary",
    "cohort": "cohort_retention",
    "retention": "cohort_retention",
    "forecast": "forecast_summary",
    "future": "forecast_summary",
    "prediction": "forecast_summary",
    "ab": "ab_test_results",
    "variant": "ab_test_results",
    "markov": "removal_effect",
    "attribution": "removal_effect",
}

DEFAULT_TABLE = "campaign_summary"
_table_cache: dict[str, pd.DataFrame] = {}


def load_table(table_name: str) -> pd.DataFrame:
    """Load a table from DB, using an in-memory cache to avoid repeat queries."""
    if table_name not in _table_cache:
        query = text(f"SELECT * FROM {table_name}")
        _table_cache[table_name] = pd.read_sql(query, engine)
    return _table_cache[table_name]


def get_required_tables(question: str) -> list[str]:
    """Match keywords in the question to relevant tables."""
    question = question.lower()
    tables = {tbl for kw, tbl in TABLE_MAPPING.items() if kw in question}
    return list(tables) or [DEFAULT_TABLE]


def load_context(question: str, rows: int = 15) -> str:
    """Build markdown context blocks for all relevant tables."""
    tables = get_required_tables(question)
    return "\n\n".join(
        f"### {table}\n{load_table(table).head(rows).to_markdown(index=False)}"
        for table in tables
    )


def ask_ai(question: str) -> str:
    context = load_context(question)
    prompt = f"""You are a Senior Marketing Data Analyst.

Answer ONLY using the information provided below.
If the answer cannot be determined from the data, clearly state that there is insufficient information.

=========================
DATABASE
=========================
{context}

=========================
QUESTION
=========================
{question}

Provide:
1. Direct Answer
2. Explanation
3. Business Recommendation
"""
    return model.generate_content(prompt).text


def main():
    print("=" * 80)
    print("AI MARKETING ANALYST")
    print("=" * 80)
    print("Type 'exit' to quit.\n")

    while True:
        question = input("Ask AI > ").strip()
        if question.lower() == "exit":
            break
        if not question:
            continue

        print(f"\n{ask_ai(question)}\n{'-' * 80}")


if __name__ == "__main__":
    main()