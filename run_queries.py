"""Run every query in queries.sql against palm.db and print the results.
Run:  python build_database.py   (once)   then   python run_queries.py"""
import re, sqlite3
import pandas as pd

con = sqlite3.connect("palm.db")
text = open("queries.sql", encoding="utf-8").read()
for name, sql in re.findall(r"-- name: (.+?)\n(.*?)(?=\n-- name:|\Z)", text, flags=re.S):
    print("\n==", name.strip(), "==")
    print(pd.read_sql_query(sql.strip(), con).to_string(index=False))
