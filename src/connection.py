import psycopg2

from dotenv import load_dotenv
import os

load_dotenv()  # take environment variables from .env.

postgres_key = os.getenv('')
ENV = os.getenv('ENVIRONMENT')

host = '' if ENV == 'production' else ''

def create_conn():
    try:
        conn = psycopg2.connect(
            dbname="", 
            user="", 
            password=postgres_key, 
            host=host,
            port=""
        )
        print("Database connection successful")
        return conn
    except Exception as e:
        print(f"An error occurred when connecting to the database: {e}")
        return None
    
def get_report(conn, report_id):
    cur = conn.cursor()
    cur.execute("SELECT * FROM your_table_name WHERE id = %s", (report_id,))
    rows = cur.fetchall()
    cur.close()
    return rows