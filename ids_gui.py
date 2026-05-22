"""
IDS 2017 Streaming Demo — Streamlit GUI
Loads CIC-IDS-2017 cleaned parquet, replays rows through Model 4 (Spark RF),
shows live predictions vs ground truth.

Run: streamlit run ids_gui.py
"""

import os
import time
import threading
import streamlit as st
import pandas as pd
import numpy as np

os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-11-openjdk"
os.environ["PYSPARK_PYTHON"] = (
    "/home/varunadhityagb/localProjects/BigData-Intrusion/.venv/bin/python3"
)
os.environ["PYSPARK_DRIVER_PYTHON"] = (
    "/home/varunadhityagb/localProjects/BigData-Intrusion/.venv/bin/python3"
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="IDS Live Monitor",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Exo+2:wght@300;600;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Exo 2', sans-serif;
    background-color: #050d1a;
    color: #c8d8e8;
}
.stApp { background-color: #050d1a; }

h1, h2, h3 { font-family: 'Share Tech Mono', monospace; color: #00e5ff; letter-spacing: 2px; }

.metric-card {
    background: linear-gradient(135deg, #0a1628 0%, #0d2040 100%);
    border: 1px solid #1a3a5c;
    border-left: 3px solid #00e5ff;
    border-radius: 4px;
    padding: 16px 20px;
    margin-bottom: 10px;
}
.metric-value { font-family: 'Share Tech Mono', monospace; font-size: 2rem; color: #00e5ff; }
.metric-label { font-size: 0.75rem; color: #6a8aaa; text-transform: uppercase; letter-spacing: 1px; }

.alert-row {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.78rem;
    padding: 4px 10px;
    margin: 2px 0;
    border-radius: 2px;
    border-left: 3px solid;
}
.alert-BENIGN    { border-color: #1a3a5c; color: #4a7a9a; background: #060e1a; }
.alert-DDOS      { border-color: #ff3860; color: #ff6b80; background: #1a0610; }
.alert-DOS       { border-color: #ff6b35; color: #ff9060; background: #1a0e06; }
.alert-BOT       { border-color: #ffdd57; color: #ffe88a; background: #1a1806; }
.alert-FTP       { border-color: #9b59b6; color: #c080e0; background: #120a1a; }
.alert-SSH       { border-color: #3498db; color: #70b8f0; background: #060e1a; }
.alert-INFIL     { border-color: #e74c3c; color: #f08070; background: #1a0806; }
.alert-WEB       { border-color: #e67e22; color: #f0a050; background: #1a1006; }
.alert-UNKNOWN   { border-color: #555; color: #888; background: #0a0a0a; }

.status-dot-green { color: #00ff88; font-size: 0.8rem; }
.status-dot-red   { color: #ff3860; font-size: 0.8rem; }

div[data-testid="stSidebar"] { background-color: #030a14; border-right: 1px solid #0d2040; }
.stSlider > div { color: #00e5ff; }
button[kind="primary"] {
    background: #00e5ff !important;
    color: #050d1a !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-weight: bold !important;
    border: none !important;
}
button[kind="secondary"] {
    background: transparent !important;
    color: #00e5ff !important;
    border: 1px solid #00e5ff !important;
    font-family: 'Share Tech Mono', monospace !important;
}
</style>
""",
    unsafe_allow_html=True,
)

# ── Constants ─────────────────────────────────────────────────────────────────
HDFS_BASE = "hdfs://lattitude7420:9000"
DATA_PATH = f"{HDFS_BASE}/ids2017/cleaned"
MODEL_PATH = f"{HDFS_BASE}/ids2018/models/multiclass_weighted_v2"
LABEL_COL = "Label"  # ground truth col in 2017 cleaned parquet

CLASS_COLORS = {
    "Benign": "#4a7a9a",
    "DDoS": "#ff3860",
    "DoS": "#ff6b35",
    "Bot": "#ffdd57",
    "FTP-BruteForce": "#9b59b6",
    "SSH-Bruteforce": "#3498db",
    "Infilteration": "#e74c3c",
    "WebAttack": "#e67e22",
}


# ── Spark init (cached) ───────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Starting Spark session...")
def get_spark():
    from pyspark.sql import SparkSession

    spark = (
        SparkSession.builder.appName("IDS-StreamingGUI")
        .master("spark://lattitude7420:7077")
        .config("spark.executor.memory", "4g")
        .config("spark.driver.memory", "4g")
        .config("spark.executor.cores", "2")
        .config("spark.memory.fraction", "0.6")
        .config("spark.memory.storageFraction", "0.3")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    return spark


@st.cache_resource(show_spinner="Loading Model 4 from HDFS...")
def get_model():
    from pyspark.ml import PipelineModel

    spark = get_spark()
    return PipelineModel.load(MODEL_PATH)


@st.cache_resource(show_spinner="Loading 2017 data from HDFS...")
def get_data_sample(n_rows: int = 50000):
    """Load a sample of 2017 data, shuffle, return as pandas."""
    spark = get_spark()
    df = spark.read.parquet(DATA_PATH)
    # sample deterministically for reproducible demo
    sample = df.sample(fraction=min(n_rows / 2_830_743, 1.0), seed=42).limit(n_rows)
    return sample.toPandas()


# ── Prediction helper ─────────────────────────────────────────────────────────
def predict_batch(pdf: pd.DataFrame) -> pd.DataFrame:
    """Run a pandas batch through Spark model, return df with 'predicted' col."""
    spark = get_spark()
    model = get_model()
    sdf = spark.createDataFrame(pdf)
    preds = model.transform(sdf)
    # Get label→name mapping from fitted StringIndexer (stage 2)
    indexer = model.stages[2]
    labels = indexer.labels  # index→name list
    pred_pd = preds.select("prediction").toPandas()
    pred_pd["predicted"] = pred_pd["prediction"].apply(
        lambda i: labels[int(i)] if int(i) < len(labels) else "Unknown"
    )
    return pred_pd["predicted"].values


# ── Session state init ────────────────────────────────────────────────────────
def init_state():
    defaults = {
        "running": False,
        "cursor": 0,
        "total_seen": 0,
        "correct": 0,
        "class_counts": {},
        "alert_log": [],  # list of dicts
        "data": None,
        "batch_size": 50,
        "sleep_ms": 500,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_state()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ CONFIG")
    st.markdown("---")

    batch_size = st.slider("Batch size (rows/tick)", 10, 500, 50, 10)
    sleep_ms = st.slider("Tick interval (ms)", 100, 3000, 500, 100)
    n_sample = st.selectbox(
        "Dataset sample size", [10_000, 25_000, 50_000, 100_000], index=2
    )

    st.markdown("---")
    load_btn = st.button(
        "📂 Load Data + Model", use_container_width=True, type="primary"
    )
    start_btn = st.button("▶ START", use_container_width=True, type="primary")
    stop_btn = st.button("⏹ STOP", use_container_width=True, type="secondary")
    reset_btn = st.button("↺ RESET", use_container_width=True, type="secondary")

    st.markdown("---")
    st.markdown("**Model:** `multiclass_weighted_v2`")
    st.markdown("**Data:** CIC-IDS-2017")
    st.markdown(f"**Spark:** `spark://lattitude7420:7077`")

    if st.session_state.data is not None:
        st.markdown(f"**Loaded:** `{len(st.session_state.data):,}` rows")
        st.markdown(
            '<span class="status-dot-green">● READY</span>', unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<span class="status-dot-red">● NOT LOADED</span>', unsafe_allow_html=True
        )

# ── Button logic ──────────────────────────────────────────────────────────────
if load_btn:
    with st.spinner("Loading data and model..."):
        st.session_state.data = get_data_sample(n_sample)
        get_model()  # warm up
    st.success(f"Loaded {len(st.session_state.data):,} rows. Model ready.")

if start_btn and st.session_state.data is not None:
    st.session_state.running = True

if stop_btn:
    st.session_state.running = False

if reset_btn:
    for k in [
        "cursor",
        "total_seen",
        "correct",
        "class_counts",
        "alert_log",
        "running",
    ]:
        if k in ["class_counts"]:
            st.session_state[k] = {}
        elif k in ["alert_log"]:
            st.session_state[k] = []
        elif k in ["running"]:
            st.session_state[k] = False
        else:
            st.session_state[k] = 0

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("# 🛡️ IDS LIVE MONITOR")
st.markdown(
    f"**Dataset:** CIC-IDS-2017 &nbsp;|&nbsp; "
    f"**Model:** Weighted RF (capped) &nbsp;|&nbsp; "
    f"**Status:** {'<span style=\"color:#00ff88\">● RUNNING</span>' if st.session_state.running else '<span style=\"color:#ff3860\">● STOPPED</span>'}",
    unsafe_allow_html=True,
)
st.markdown("---")

# ── Main layout ───────────────────────────────────────────────────────────────
col_metrics, col_chart, col_feed = st.columns([1.2, 2, 1.8])

# Metrics
with col_metrics:
    st.markdown("### STATS")
    total = st.session_state.total_seen
    acc = (st.session_state.correct / total * 100) if total > 0 else 0.0
    alerts = sum(v for k, v in st.session_state.class_counts.items() if k != "Benign")

    for label, value in [
        ("Flows Processed", f"{total:,}"),
        ("Accuracy", f"{acc:.1f}%"),
        ("Alerts (Non-Benign)", f"{alerts:,}"),
    ]:
        st.markdown(
            f"""
        <div class="metric-card">
            <div class="metric-value">{value}</div>
            <div class="metric-label">{label}</div>
        </div>""",
            unsafe_allow_html=True,
        )

    st.markdown("### CLASS COUNTS")
    if st.session_state.class_counts:
        counts_df = pd.DataFrame(
            list(st.session_state.class_counts.items()), columns=["Class", "Count"]
        ).sort_values("Count", ascending=False)
        st.dataframe(counts_df, hide_index=True, use_container_width=True, height=250)

# Chart
with col_chart:
    st.markdown("### PREDICTION DISTRIBUTION")
    if st.session_state.class_counts:
        chart_df = pd.DataFrame(
            list(st.session_state.class_counts.items()), columns=["Class", "Count"]
        ).sort_values("Count", ascending=False)
        st.bar_chart(chart_df.set_index("Class"), use_container_width=True, height=300)

    st.markdown("### ACCURACY OVER TIME")
    if "acc_history" not in st.session_state:
        st.session_state.acc_history = []
    if len(st.session_state.acc_history) > 1:
        acc_df = pd.DataFrame(st.session_state.acc_history, columns=["Accuracy"])
        st.line_chart(acc_df, use_container_width=True, height=180)

# Alert feed
with col_feed:
    st.markdown("### ALERT FEED")
    feed_html = ""
    # Show last 30 alerts, newest first
    for entry in reversed(st.session_state.alert_log[-30:]):
        pred = entry["predicted"]
        truth = entry["truth"]
        ok = "✓" if pred == truth else "✗"
        cls_key = pred.replace("-", "").replace(" ", "").upper()[:5]
        css_cls = {
            "Benign": "BENIGN",
            "DDoS": "DDOS",
            "DoS": "DOS",
            "Bot": "BOT",
            "FTP-BruteForce": "FTP",
            "SSH-Bruteforce": "SSH",
            "Infilteration": "INFIL",
            "WebAttack": "WEB",
        }.get(pred, "UNKNOWN")
        feed_html += (
            f'<div class="alert-row alert-{css_cls}">{ok} {pred:<18} GT:{truth}</div>'
        )
    st.markdown(feed_html, unsafe_allow_html=True)

# ── Streaming tick ────────────────────────────────────────────────────────────
if st.session_state.running and st.session_state.data is not None:
    data = st.session_state.data
    cursor = st.session_state.cursor

    if cursor >= len(data):
        st.session_state.running = False
        st.info("Replay complete.")
    else:
        end = min(cursor + batch_size, len(data))
        batch = data.iloc[cursor:end].copy()

        # drop label cols before inference
        infer_batch = batch.drop(
            columns=[
                c
                for c in [LABEL_COL, "attack_type", "binary_label"]
                if c in batch.columns
            ]
        )

        try:
            predictions = predict_batch(infer_batch)
        except Exception as e:
            st.error(f"Prediction error: {e}")
            st.session_state.running = False
            predictions = []

        if len(predictions) > 0:
            truths = (
                batch[LABEL_COL].fillna("Unknown").values
                if LABEL_COL in batch.columns
                else ["Unknown"] * len(predictions)
            )

            for pred, truth in zip(predictions, truths):
                # normalize truth label for comparison (2017 Label col has raw strings)
                st.session_state.total_seen += 1
                if pred == truth or truth.strip().upper() in pred.upper():
                    st.session_state.correct += 1
                st.session_state.class_counts[pred] = (
                    st.session_state.class_counts.get(pred, 0) + 1
                )
                if pred != "Benign":
                    st.session_state.alert_log.append(
                        {"predicted": pred, "truth": truth.strip()}
                    )

            # track accuracy history (one point per batch)
            total = st.session_state.total_seen
            acc = st.session_state.correct / total * 100 if total > 0 else 0
            st.session_state.acc_history.append(acc)
            if len(st.session_state.acc_history) > 200:
                st.session_state.acc_history = st.session_state.acc_history[-200:]

        st.session_state.cursor += batch_size
        time.sleep(sleep_ms / 1000.0)
        st.rerun()
