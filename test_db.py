import os
import certifi
import psycopg
from dotenv import load_dotenv

load_dotenv()

os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

database_url = os.getenv("DATABASE_URL")

if not database_url:
    raise ValueError("DATABASE_URL missing")

if "sslmode=" not in database_url.lower():
    database_url += "&sslmode=require" if "?" in database_url else "?sslmode=require"

print("Connecting to PostgreSQL...")

try:

    with psycopg.connect(
        database_url,
        connect_timeout=20
    ) as conn:

        with conn.cursor() as cur:

            cur.execute("SELECT version();")

            result = cur.fetchone()

            print("\n✅ PostgreSQL connection successful!")
            print(result[0])

except Exception as e:

    print("\n❌ PostgreSQL connection failed!")
    print(type(e).__name__)
    print(e)