# ==========================================================
# AI BUSINESS INSIGHTS USING GEMINI
# ==========================================================

import logging
import os

import google.generativeai as genai
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import text

from db_connection import engine

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

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
# TABLES TO ANALYZE
# Maps a friendly module name -> the actual table saved by each upstream script
# ==========================================================

ANALYSIS_TABLES = {
    "campaign_summary": "campaign_summary",
    "funnel_summary": "funnel_summary",
    "lifecycle_summary": "lifecycle_summary",
    "cohort_retention": "cohort_retention",
    "ab_test_results": "ab_test_results",
    "removal_effect": "markov_removal_effect",
    "forecast_summary": "forecast_summary",
}

PROMPT_TEMPLATES = {
    "campaign_summary": """You are a Senior Marketing Analyst.

Analyze the campaign performance below.

Tasks:
1. Identify the best campaigns.
2. Identify the weakest campaigns.
3. Explain why.
4. Recommend budget allocation.

Data:

{data}
""",
    "funnel_summary": """You are a CRO Specialist.

Analyze the funnel below.

Tasks:
1. Largest drop-off
2. Possible reasons
3. Suggestions to improve conversion

Data:

{data}
""",
    "lifecycle_summary": """You are a Customer Analytics Expert.

Analyze customer segments.

Tasks:
1. Best customer segments
2. Lowest performing segments
3. Marketing recommendations

Data:

{data}
""",
    "cohort_retention": """Analyze customer retention.

Identify:
1. Strongest cohort
2. Weakest cohort
3. Retention trend
4. Retention strategy

Data:

{data}
""",
    "ab_test_results": """Analyze the A/B Testing results.

Identify:
1. Winning Variant
2. Significant Results
3. Deployment Recommendation

Data:

{data}
""",
    "removal_effect": """Analyze the Markov Attribution.

Identify:
1. Most important touchpoints
2. Budget recommendation
3. Attribution insight

Data:

{data}
""",
    "forecast_summary": """Analyze the forecast.

Identify:
1. Revenue Trend
2. Risks
3. Growth Opportunities

Data:

{data}
""",
}


def load_table(table_name):
    """Load a single table from MySQL."""
    logging.info(f"Loading {table_name}...")
    return pd.read_sql(text(f"SELECT * FROM {table_name}"), engine)


def load_analysis_tables():
    """Load every upstream analysis table needed for insight generation."""
    logging.info("Loading Analysis Tables...")

    tables = {module_name: load_table(table_name) for module_name, table_name in ANALYSIS_TABLES.items()}

    logging.info("All Analysis Tables Loaded.")
    return tables


def validate_tables(tables):
    """Print row counts for each loaded table as a sanity check."""
    logging.info("Validating Loaded Tables...")

    for table_name, df in tables.items():
        print(f"{table_name:<25}{len(df):>8} rows")

    logging.info("Validation Completed.")


def create_prompt(module_name, dataframe):
    """Build the analysis prompt for a given module, using up to 20 sample rows."""
    data = dataframe.head(20).to_markdown(index=False)
    return PROMPT_TEMPLATES[module_name].format(data=data)


def generate_ai_insight(module_name, dataframe):
    """Generate a single AI-written insight for one table."""
    logging.info(f"Generating Insight : {module_name}")

    prompt = create_prompt(module_name, dataframe)
    response = model.generate_content(prompt)

    return response.text


def generate_all_insights(tables):
    """Generate AI insights for every loaded table."""
    insights = {table_name: generate_ai_insight(table_name, df) for table_name, df in tables.items()}

    logging.info("All AI Insights Generated.")
    return insights


def ask_ai(question, context):
    """Ask a free-form question grounded only in the given context."""
    prompt = f"""You are an AI Marketing Analyst.

Use ONLY the information provided below.

If the answer cannot be determined from the data,
say that the available data is insufficient.

==========================
DATABASE
==========================

{context}

==========================
QUESTION
==========================

{question}

Provide:
1. Direct Answer
2. Explanation
3. Business Recommendation
"""

    response = model.generate_content(prompt)
    return response.text


def save_insights(insights):
    """Persist generated insights to MySQL and CSV."""
    logging.info("Saving AI Insights...")

    df = pd.DataFrame({"Module": insights.keys(), "Insight": insights.values()})
    df.to_sql("ai_insights", engine, if_exists="replace", index=False)
    df.to_csv("ai_insights.csv", index=False)

    logging.info("AI Insights Saved.")


def display_insights(insights):
    """Print every generated insight."""
    print("\n" + "=" * 100)
    print("AI GENERATED BUSINESS INSIGHTS")
    print("=" * 100)

    for module, insight in insights.items():
        print(f"\n{module.upper()}")
        print("-" * 100)
        print(insight)
        print()


def main():
    try:
        tables = load_analysis_tables()
        validate_tables(tables)

        insights = generate_all_insights(tables)

        save_insights(insights)
        display_insights(insights)

        logging.info("AI Insight Generation Completed.")

    except Exception as e:
        logging.exception(e)


if __name__ == "__main__":
    main()