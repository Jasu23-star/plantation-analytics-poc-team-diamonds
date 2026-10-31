"""Palm Oil Yield Optimization Analytics POC - Streamlit dashboard (Team Diamonds).

Run:  streamlit run app.py
Data: SOPB_Palm_Yield_Optimization_POC_SAC.csv (synthetic sample data) in the same folder.
"""
import re
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold, cross_val_predict

st.set_page_config(page_title="Palm Yield Analytics POC", layout="wide")
Y = "ffb_yield_t_ha"
CSV = Path(__file__).parent / "SOPB_Palm_Yield_Optimization_POC_SAC.csv"
GREEN, AMBER, RED, GREY = "#1F5C3A", "#E0A030", "#B5432F", "#9AA79E"

AGE_ORDER = ["Early Young (4-5y)", "Young (6-7y)", "Late Young (8-10y)", "Early Prime (11-13y)",
             "Prime (14-16y)", "Late Prime (17-20y)", "Old (21-24y)", "Replant Candidate (25y+)"]
FACTORS = {
    "Harvest interval (days)": "harvest_interval_days",
    "Water deficit, prior 90 days (mm)": "water_deficit_prior_90d_mm",
    "Rainfall, prior 90 days (mm)": "rainfall_prior_90d_mm",
    "Rainfall this cycle (mm)": "rainfall_cycle_mm",
    "Stand density (palms/ha)": "stand_density_palms_ha",
    "Missing palms (%)": "missing_palm_pct",
    "Ganoderma incidence (%)": "ganoderma_incidence_pct",
    "Leaf potassium (%)": "leaf_k_pct",
    "Leaf magnesium (%)": "leaf_mg_pct",
    "Soil pH": "soil_ph",
    "Road accessibility (1-5, 5 best)": "road_accessibility_score_1_5",
    "Bagworm incidence (%)": "bagworm_incidence_pct",
    "Fertilizer applied (kg/ha)": "fertilizer_total_kg_ha",
    "Weed cover (%)": "weed_cover_pct",
}


def wyield(d):
    """Area-weighted yield: total tonnes / total productive hectares (never sum yields)."""
    return d.ffb_tonnes.sum() / d.productive_area_ha.sum()


@st.cache_data
def load():
    df = pd.read_csv(CSV, parse_dates=["harvest_end_date"])
    df["yield_14d"] = df[Y] / df.harvest_interval_days * 14          # fair comparison across round lengths
    df["age_expected"] = df.groupby("palm_age_years")[Y].transform("mean")  # what age alone predicts
    df["age_adj"] = df[Y] - df.age_expected                            # yield after removing the age effect
    return df


@st.cache_data
def block_table(df):
    b = df.groupby(["estate_name", "division_name", "block_name", "block_id"]).agg(
        area_ha=("productive_area_ha", "first"), age=("palm_age_years", "first"),
        yield_t_ha=(Y, "mean"), expected=("age_expected", "mean"),
        road=("road_accessibility_score_1_5", "mean"), water_deficit=("water_deficit_prior_90d_mm", "mean"),
        ganoderma=("ganoderma_incidence_pct", "mean"), missing_palms=("missing_palm_pct", "mean"),
        lat=("block_latitude", "first"), lon=("block_longitude", "first")).reset_index()
    b["gap_pct"] = (b.yield_t_ha / b.expected - 1) * 100
    b["t_lost_per_cycle"] = -(b.yield_t_ha - b.expected) * b.area_ha

    def driver(r):
        if r.age >= 25: return "Replant candidate"
        if r.water_deficit > 20: return "Water stress"
        if r.road < 2.3: return "Poor access / late harvest"
        if r.missing_palms > 5: return "Missing palms"
        if r.ganoderma > 1.2: return "Ganoderma"
        return "Mixed factors"
    b["likely_driver"] = b.apply(driver, axis=1)
    return b


@st.cache_resource
def train(df):
    cat = pd.get_dummies(df[["soil_type", "drainage_class", "genetic_line", "fertilizer_program",
                             "treatment_program"]]).astype(float)
    drop = {"record_id", "ffb_tonnes", Y, "yield_14d", "age_expected", "age_adj", "bunch_count",
            "avg_bunch_weight_kg", "ripe_bunch_pct", "unripe_bunch_pct", "overripe_bunch_pct",
            "empty_abnormal_bunch_pct", "loose_fruit_kg_ha", "estimated_field_loss_pct",
            "mesocarp_to_fruit_pct", "oil_to_bunch_pct", "kernel_to_bunch_pct", "ffa_pct",
            "estate_latitude", "estate_longitude", "block_latitude", "block_longitude",
            "harvest_cycle_no", "productive_palms_count", "planting_year"}
    num = [c for c in df.select_dtypes("number").columns if c not in drop]
    X = pd.concat([df[num].fillna(df[num].median()), cat], axis=1)
    model = GradientBoostingRegressor(n_estimators=250, max_depth=3, learning_rate=0.05,
                                      subsample=0.8, random_state=1)
    # Validate on whole blocks the model has never seen (5-fold, grouped by block).
    pred = cross_val_predict(model, X, df[Y], cv=GroupKFold(5), groups=df.block_id)
    model.fit(X, df[Y])
    imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    return pred, imp


@st.cache_resource
def sql_connection():
    """In-memory SQLite database holding the same data in a table called harvest_records."""
    con = sqlite3.connect(":memory:", check_same_thread=False)
    raw = pd.read_csv(CSV)
    raw.to_sql("harvest_records", con, index=False)
    return con


def load_queries():
    text = (Path(__file__).parent / "queries.sql").read_text(encoding="utf-8")
    return {n.strip(): q.strip() for n, q in re.findall(r"-- name: (.+?)\n(.*?)(?=\n-- name:|\Z)", text, flags=re.S)}


df = load()
blocks = block_table(df)

# ---------------- Sidebar filters ----------------
st.sidebar.title("Filters")
estates = st.sidebar.multiselect("Estate", sorted(df.estate_name.unique()), default=sorted(df.estate_name.unique()))
ages = st.sidebar.multiselect("Palm age band", AGE_ORDER, default=AGE_ORDER)
f = df[df.estate_name.isin(estates) & df.palm_age_band.isin(ages)]
fb = blocks[blocks.estate_name.isin(estates) & blocks.block_id.isin(f.block_id.unique())]
st.sidebar.caption("Synthetic sample data (SOPB POC dataset). Not actual plantation results.")

st.title("Palm Oil Yield Optimization Analytics POC")
st.caption("From 'what happened' to 'why' and 'what to do next'  |  Team Diamonds")
if f.empty:
    st.warning("No data for this filter selection.")
    st.stop()

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["1. Overview", "2. Why yield differs", "3. Hotspots & actions",
                                              "4. Forecast", "5. Harvest quality", "6. SQL"])

# ---------------- 1. Overview ----------------
with tab1:
    c = st.columns(4)
    c[0].metric("Total FFB harvested", f"{f.ffb_tonnes.sum():,.0f} t")
    c[1].metric("Area-weighted yield", f"{wyield(f):.3f} t/ha per cycle")
    c[2].metric("Blocks", f"{f.block_id.nunique():,}")
    c[3].metric("Harvest records", f"{len(f):,}")
    left, right = st.columns(2)
    est = f.groupby("estate_name").apply(wyield).sort_values().rename("Yield (t/ha per cycle)").reset_index()
    left.subheader("Yield by estate")
    left.plotly_chart(px.bar(est, x="Yield (t/ha per cycle)", y="estate_name", orientation="h",
                             color_discrete_sequence=[GREEN]).update_layout(yaxis_title=None, height=380),
                      width="stretch")
    heat = f.groupby(["estate_name", "division_name"]).apply(wyield).unstack()
    right.subheader("Yield by estate and division")
    right.plotly_chart(px.imshow(heat, color_continuous_scale="YlGn", aspect="auto",
                                 labels=dict(color="t/ha")).update_layout(height=380), width="stretch")
    st.subheader("Drill down: blocks in one estate")
    pick = st.selectbox("Estate", sorted(f.estate_name.unique()))
    bb = fb[fb.estate_name == pick].sort_values("yield_t_ha")
    st.plotly_chart(px.bar(bb, x="block_name", y="yield_t_ha", color="gap_pct", color_continuous_scale="RdYlGn",
                           labels={"yield_t_ha": "Yield (t/ha per cycle)", "gap_pct": "vs age-expected (%)"},
                           hover_data=["division_name", "age"]).update_layout(height=360, xaxis_title=None),
                    width="stretch")

# ---------------- 2. Why yield differs ----------------
with tab2:
    st.markdown("Palm age is the biggest driver, so factors are also shown **after removing the age effect**.")
    a, b_ = st.columns(2)
    age = f.groupby("palm_age_band")[Y].mean().reindex(AGE_ORDER).dropna().reset_index()
    a.subheader("Yield by palm age")
    a.plotly_chart(px.bar(age, x="palm_age_band", y=Y, color_discrete_sequence=[GREEN],
                          labels={Y: "t/ha per cycle", "palm_age_band": ""}).update_layout(height=360),
                   width="stretch")
    f2 = f.assign(wd=pd.cut(f.water_deficit_prior_90d_mm, [-1, 0, 20, 60, 1000],
                            labels=["None", "1-20 mm", "21-60 mm", ">60 mm"]))
    wd = f2.groupby("wd", observed=True).yield_14d.agg(["mean", "size"]).reset_index()
    b_.subheader("Water deficit in the prior 90 days")
    b_.plotly_chart(px.bar(wd, x="wd", y="mean", text="size", color_discrete_sequence=[AMBER],
                           labels={"mean": "Yield per 14-day cycle (t/ha)", "wd": ""}).update_layout(height=360),
                    width="stretch")
    b_.caption("Numbers on bars = records. Compared on an equal 14-day cycle so harvest-round length does not distort it.")
    st.subheader("Which factors move with yield? (age effect removed)")
    cors = pd.Series({k: f[v].corr(f.age_adj) for k, v in FACTORS.items()}).sort_values()
    st.plotly_chart(px.bar(cors, orientation="h", color=cors.values, color_continuous_scale="RdYlGn",
                           labels={"value": "Correlation with age-adjusted yield", "index": ""})
                    .update_layout(height=440, showlegend=False, coloraxis_showscale=False), width="stretch")
    st.caption("Correlation is not causation. Harvest interval is largely mechanical: longer rounds simply collect more fruit per cycle.")
    st.subheader("Explore one factor")
    name = st.selectbox("Factor", list(FACTORS))
    col = FACTORS[name]
    g = f.assign(bin=pd.qcut(f[col], 8, duplicates="drop")).groupby("bin", observed=True).agg(
        x=(col, "mean"), y=("age_adj", "mean")).reset_index()
    st.plotly_chart(px.line(g, x="x", y="y", markers=True, labels={"x": name, "y": "Yield vs age-expected (t/ha)"},
                            color_discrete_sequence=[GREEN]).update_layout(height=340), width="stretch")

# ---------------- 3. Hotspots & actions ----------------
with tab3:
    st.markdown("Blocks ranked by tonnes lost against what their palm age should produce.")
    m1, m2, m3 = st.columns(3)
    low = fb[fb.gap_pct < fb.gap_pct.quantile(0.25)]
    upside = ((fb.gap_pct.median() - low.gap_pct) / 100 * low.expected * low.area_ha).sum()
    m1.metric("Blocks >10% below age-expected", int((fb.gap_pct < -10).sum()))
    m2.metric("Replant candidates (25y+)", int((fb.age >= 25).sum()))
    m3.metric("Indicative upside if the weakest quarter reaches the median", f"{upside:,.0f} t per cycle")
    mapfn = getattr(px, "scatter_map", None) or px.scatter_mapbox
    kw = {} if hasattr(px, "scatter_map") else {"mapbox_style": "open-street-map"}
    fig = mapfn(fb, lat="lat", lon="lon", color="gap_pct", color_continuous_scale="RdYlGn", range_color=(-20, 20),
                hover_name="block_id", hover_data={"estate_name": True, "gap_pct": ":.1f", "likely_driver": True,
                                                   "lat": False, "lon": False}, zoom=6.3, height=460, **kw)
    st.plotly_chart(fig, width="stretch")
    st.subheader("Priority blocks")
    n = st.slider("Number of blocks", 5, 50, 15)
    top = fb.sort_values("t_lost_per_cycle", ascending=False).head(n)[
        ["block_id", "estate_name", "age", "gap_pct", "t_lost_per_cycle", "likely_driver"]].round(1)
    st.dataframe(top.rename(columns={"gap_pct": "below expected (%)", "t_lost_per_cycle": "t lost per cycle"}),
                 width="stretch", hide_index=True)
    st.download_button("Download priority list (CSV)", top.to_csv(index=False), "priority_blocks.csv")
    st.caption("'Likely driver' is a simple rule-based flag to guide an agronomist's visit, not a proven cause.")

# ---------------- 4. Forecast ----------------
with tab4:
    with st.spinner("Training the model and validating it on unseen blocks..."):
        pred, imp = train(df)
    naive_mae = mean_absolute_error(df[Y], df.prior_cycle_yield_t_ha)
    mae = mean_absolute_error(df[Y], pred)
    k = st.columns(3)
    k[0].metric("R² on unseen blocks", f"{r2_score(df[Y], pred):.2f}")
    k[1].metric("Model average error", f"{mae:.3f} t/ha")
    k[2].metric("'Same as last cycle' error", f"{naive_mae:.3f} t/ha", delta=f"{(mae / naive_mae - 1) * 100:.0f}% with model",
                delta_color="inverse")
    l, r = st.columns(2)
    d = df.assign(predicted=pred)
    l.subheader("Predicted vs actual")
    l.plotly_chart(px.scatter(d.sample(1200, random_state=1), x=Y, y="predicted", opacity=0.45,
                              color_discrete_sequence=[GREEN]).update_layout(height=380), width="stretch")
    r.subheader("Strongest signals in the model")
    r.plotly_chart(px.bar(imp.head(10)[::-1], orientation="h", color_discrete_sequence=[AMBER],
                          labels={"value": "Importance", "index": ""}).update_layout(height=380, showlegend=False),
                   width="stretch")
    st.subheader("Check one block")
    blk = st.selectbox("Block", sorted(d.block_id.unique()))
    one = d[d.block_id == blk].sort_values("harvest_cycle_no")
    st.plotly_chart(px.line(one, x="harvest_cycle_no", y=[Y, "predicted"], markers=True,
                            labels={"value": "t/ha per cycle", "harvest_cycle_no": "Harvest cycle"})
                    .update_layout(height=320), width="stretch")
    st.caption("Validation holds out whole blocks (5-fold), so the model is judged on blocks it has never seen.")

# ---------------- 5. Harvest quality ----------------
with tab5:
    st.markdown("Poor field access delays harvesting, which raises overripe bunches and field losses.")
    f5 = f.assign(access=pd.cut(f.road_accessibility_score_1_5, [0, 2, 3, 5.1],
                                labels=["Poor (1-2)", "Fair (2-3)", "Good (3-5)"]))
    q = f5.groupby("access", observed=True)[["harvest_interval_days", "overripe_bunch_pct",
                                              "estimated_field_loss_pct", "ffa_pct"]].mean().reset_index()
    c1, c2 = st.columns(2)
    c1.plotly_chart(px.bar(q, x="access", y="estimated_field_loss_pct", color_discrete_sequence=[RED],
                           labels={"estimated_field_loss_pct": "Field loss (%)", "access": "Road access"},
                           title="Field loss").update_layout(height=340), width="stretch")
    c2.plotly_chart(px.bar(q, x="access", y="overripe_bunch_pct", color_discrete_sequence=[AMBER],
                           labels={"overripe_bunch_pct": "Overripe bunches (%)", "access": "Road access"},
                           title="Overripe bunches").update_layout(height=340), width="stretch")
    st.dataframe(q.round(2).rename(columns={"harvest_interval_days": "avg harvest interval (days)",
                                            "overripe_bunch_pct": "overripe (%)",
                                            "estimated_field_loss_pct": "field loss (%)", "ffa_pct": "FFA (%)"}),
                 width="stretch", hide_index=True)

# ---------------- 6. SQL ----------------
with tab6:
    st.markdown("The same findings, reproduced with **SQL** on a SQLite database (table: `harvest_records`). "
                "The queries live in `queries.sql`.")
    con = sql_connection()
    queries = load_queries()
    choice = st.selectbox("Choose a saved query", list(queries))
    st.code(queries[choice], language="sql")
    st.dataframe(pd.read_sql_query(queries[choice], con), width="stretch", hide_index=True)
    st.subheader("Write your own query (SELECT only)")
    custom = st.text_area("SQL", "SELECT estate_name, COUNT(DISTINCT block_id) AS blocks\nFROM harvest_records\nGROUP BY estate_name;", height=120)
    if st.button("Run query"):
        if not re.match(r"^\s*(select|with)\b", custom, flags=re.I) or ";" in custom.strip().rstrip(";"):
            st.error("Only a single SELECT query is allowed.")
        else:
            try:
                st.dataframe(pd.read_sql_query(custom, con).head(500), width="stretch", hide_index=True)
            except Exception as e:
                st.error(f"SQL error: {e}")
