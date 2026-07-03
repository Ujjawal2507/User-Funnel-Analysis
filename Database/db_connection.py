
# MySQL Database Connection


import os

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.engine import URL

load_dotenv()



DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "3306")
DB_NAME = os.getenv("DB_NAME", "user_journey_analysis")
DB_USER = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD", "Ujjawal@2511")

if not DB_USER or not DB_PASSWORD:
    raise RuntimeError(
        "DB_USER and DB_PASSWORD must be set. Copy .env.example to .env "
        "and fill in your MySQL credentials before running this script."
    )

# ==========================================================
# DATABASE URL
# ==========================================================

DATABASE_URL = URL.create(
    drivername="mysql+pymysql",
    username=DB_USER,
    password=DB_PASSWORD,
    host=DB_HOST,
    port=DB_PORT,
    database=DB_NAME,
)


# ==========================================================
# CREATE ENGINE
# ==========================================================

engine = create_engine(
    DATABASE_URL,
    echo=False,
    future=True
)

# ==========================================================
# TEST CONNECTION
# ==========================================================

def test_connection():
    try:
        with engine.connect():
            print("=" * 60)
            print("Successfully Connected to MySQL")
            print(f"Database : {DB_NAME}")
            print("=" * 60)

    except SQLAlchemyError as e:
        print("=" * 60)
        print(" Database Connection Failed")
        print(e)
        print("=" * 60)


if __name__ == "__main__":
    test_connection()