"""Load the CSV into a SQLite database file (palm.db) so you can query it with SQL.
Run:  python build_database.py
Open palm.db in 'DB Browser for SQLite' (free) or run queries with run_queries.py."""
import sqlite3
import pandas as pd

df = pd.read_csv("SOPB_Palm_Yield_Optimization_POC_SAC.csv")
con = sqlite3.connect("palm.db")
df.to_sql("harvest_records", con, if_exists="replace", index=False)
for col in ("block_id", "estate_name", "palm_age_years", "harvest_cycle_no"):
    con.execute(f"CREATE INDEX IF NOT EXISTS idx_{col} ON harvest_records({col})")
con.commit()
print("palm.db created:", len(df), "rows,", len(df.columns), "columns")
