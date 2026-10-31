# Palm Oil Yield Optimization Analytics POC (Team Diamonds)

Interactive Streamlit dashboard built on the synthetic SOPB sample dataset.

## Run it
1. Install Python 3.10+ from python.org
2. Open a terminal in this folder and run:
   pip install -r requirements.txt
   streamlit run app.py
3. Your browser opens at http://localhost:8501

## Pages
1. Overview: tonnes, yield by estate and division, drill down to blocks
2. Why yield differs: palm age, water deficit, factor correlations (age effect removed)
3. Hotspots and actions: map, ranked priority blocks, downloadable list
4. Forecast: model validated on blocks it has never seen
5. Harvest quality: road access vs overripe bunches and field loss
6. SQL: saved queries (queries.sql) that reproduce the findings, plus a box to run your own SELECT queries

SQL files: build_database.py makes palm.db, run_queries.py runs all queries in queries.sql.

All data is synthetic and does not represent actual plantation performance.
