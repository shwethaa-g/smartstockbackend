import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime, timedelta

USE_PROPHET = os.getenv("USE_PROPHET", "0") == "1"
HAS_PROPHET = False
if USE_PROPHET:
    try:
        from prophet import Prophet
        HAS_PROPHET = True
    except Exception as e:
        HAS_PROPHET = False

DB_PATH = os.path.join(os.path.dirname(__file__), "smartstock.db")

app = Flask(__name__)
CORS(app)

# ----------------- DB helpers -----------------
def get_conn():
    return sqlite3.connect(DB_PATH, detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES)

def init_db():
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "CREATE TABLE IF NOT EXISTS users (id TEXT PRIMARY KEY, pin TEXT NOT NULL)"
        )
        cur.execute(
            "CREATE TABLE IF NOT EXISTS inventory (product TEXT NOT NULL, batch_id TEXT, quantity INTEGER NOT NULL, expiry_date TEXT)"
        )
        cur.execute(
            "CREATE TABLE IF NOT EXISTS sales (product TEXT NOT NULL, batch_id TEXT, quantity INTEGER NOT NULL, date TEXT)"
        )
        # seed demo user if not present
        cur.execute("SELECT COUNT(*) FROM users WHERE id=?", ("demo",))
        if cur.fetchone()[0] == 0:
            cur.execute("INSERT INTO users (id, pin) VALUES (?, ?)", ("demo", "1234"))
        conn.commit()

init_db()

# --------------- Utilities --------------------
REQUIRED_INV_COLS = {"Product","BatchID","Quantity","ExpiryDate"}
REQUIRED_SALES_COLS_MIN = {"Product","Quantity","Date"}  # BatchID optional

def read_csv_to_df(file_storage):
    try:
        df = pd.read_csv(file_storage)
    except Exception:
        file_storage.stream.seek(0)
        df = pd.read_csv(file_storage, encoding="utf-8")
    return df

def normalize_inventory(df):
    df = df.rename(columns={
        "Product":"product",
        "BatchID":"batch_id",
        "Quantity":"quantity",
        "ExpiryDate":"expiry_date"
    })
    df["expiry_date"] = pd.to_datetime(df["expiry_date"], errors="coerce").dt.date
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce").fillna(0).astype(int)
    df["batch_id"] = df["batch_id"].astype(str)
    df["product"] = df["product"].astype(str)
    return df[["product","batch_id","quantity","expiry_date"]]

def normalize_sales(df):
    df = df.rename(columns={
        "Product":"product",
        "BatchID":"batch_id",
        "Quantity":"quantity",
        "Date":"date"
    })
    if "batch_id" not in df.columns:
        df["batch_id"] = None
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce").fillna(0).astype(int)
    df["product"] = df["product"].astype(str)
    return df[["product","batch_id","quantity","date"]]

def compute_current_stock():
    with get_conn() as conn:
        inv = pd.read_sql_query("SELECT * FROM inventory", conn)
        if inv.empty:
            return pd.DataFrame(columns=["product","batch_id","stock_left","expiry_date"])
        sales = pd.read_sql_query("SELECT * FROM sales", conn)

    inv = inv.copy()
    inv["quantity"] = inv["quantity"].astype(int)

    if sales.empty:
        inv["stock_left"] = inv["quantity"]
        return inv.rename(columns={"expiry_date":"expiry_date"})[["product","batch_id","stock_left","expiry_date"]]

    sales = sales.copy()
    sales["quantity"] = sales["quantity"].astype(int)

    # If batch IDs present in sales for most rows, subtract per batch
    if sales["batch_id"].notna().sum() > 0:
        sold_by_batch = sales.dropna(subset=["batch_id"]).groupby(["product","batch_id"], as_index=False)["quantity"].sum()
        stock = inv.merge(sold_by_batch, on=["product","batch_id"], how="left").rename(columns={"quantity_x":"inv_qty","quantity_y":"sold_qty"})
        stock["sold_qty"] = stock["sold_qty"].fillna(0).astype(int)
        stock["stock_left"] = (stock["inv_qty"] - stock["sold_qty"]).clip(lower=0)
        return stock[["product","batch_id","stock_left","expiry_date"]]

    # Otherwise, subtract sales at product-level (no batch info)
    sold_by_product = sales.groupby(["product"], as_index=False)["quantity"].sum().rename(columns={"quantity":"sold_qty"})
    inv_agg = inv.groupby(["product"], as_index=False).agg(
        total_qty=("quantity","sum"),
        expiry_date=("expiry_date","min")
    )
    stock = inv_agg.merge(sold_by_product, on="product", how="left")
    stock["sold_qty"] = stock["sold_qty"].fillna(0).astype(int)
    stock["stock_left"] = (stock["total_qty"] - stock["sold_qty"]).clip(lower=0)
    stock["batch_id"] = None
    return stock[["product","batch_id","stock_left","expiry_date"]]

# ----------------- Routes ---------------------
@app.route("/api/health")
def health():
    return jsonify({"ok": True, "message":"SmartStock backend running"})

@app.route("/api/login", methods=["POST"])
def login():
    data = request.get_json(force=True)
    user_id = str(data.get("user_id","")).strip()
    pin = str(data.get("pin","")).strip()
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT pin FROM users WHERE id=?", (user_id,))
        row = cur.fetchone()
    if row and row[0] == pin:
        return jsonify({"ok": True, "token": f"demo-token-{user_id}"})
    return jsonify({"ok": False, "error":"Invalid credentials"}), 401

@app.route("/api/upload/inventory", methods=["POST"])
def upload_inventory():
    if "file" not in request.files:
        return jsonify({"ok": False, "error":"No file field 'file' found"}), 400
    f = request.files["file"]
    df = read_csv_to_df(f)
    if not REQUIRED_INV_COLS.issubset(set(df.columns)):
        missing = list(REQUIRED_INV_COLS - set(df.columns))
        return jsonify({"ok": False, "error": f"Missing required columns: {missing}"}), 400
    df = normalize_inventory(df)
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("DELETE FROM inventory")
        conn.commit()
        df.to_sql("inventory", conn, if_exists="append", index=False)
    return jsonify({"ok": True, "rows_ingested": int(len(df)), "required_columns": list(REQUIRED_INV_COLS)})

@app.route("/api/upload/sales", methods=["POST"])
def upload_sales():
    if "file" not in request.files:
        return jsonify({"ok": False, "error":"No file field 'file' found"}), 400
    f = request.files["file"]
    df = read_csv_to_df(f)
    if not REQUIRED_SALES_COLS_MIN.issubset(set(df.columns)):
        missing = list(REQUIRED_SALES_COLS_MIN - set(df.columns))
        return jsonify({"ok": False, "error": f"Missing required columns: {missing}"}), 400
    df = normalize_sales(df)
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("DELETE FROM sales")
        conn.commit()
        df.to_sql("sales", conn, if_exists="append", index=False)
    return jsonify({"ok": True, "rows_ingested": int(len(df)), "required_columns": list(REQUIRED_SALES_COLS_MIN)})

@app.route("/api/stock")
def get_stock():
    stock_df = compute_current_stock()
    records = []
    for _, r in stock_df.iterrows():
        records.append({
            "product": r["product"],
            "batch_id": None if pd.isna(r["batch_id"]) else str(r["batch_id"]),
            "stock_left": int(r["stock_left"]),
            "expiry_date": None if pd.isna(r["expiry_date"]) else str(r["expiry_date"])
        })
    return jsonify({"ok": True, "data": records})

@app.route("/api/alerts")
def get_alerts():
    low_threshold = int(request.args.get("low_threshold", 5))
    expiry_days = int(request.args.get("expiry_days", 5))
    today = datetime.utcnow().date()

    stock_df = compute_current_stock()
    alerts = []

    for _, r in stock_df.iterrows():
        stock_left = int(r["stock_left"])
        if stock_left <= 0:
            alerts.append({
                "type":"out_of_stock",
                "product": r["product"],
                "batch_id": None if pd.isna(r["batch_id"]) else str(r["batch_id"])
            })
        elif stock_left <= low_threshold:
            alerts.append({
                "type":"low_stock",
                "product": r["product"],
                "batch_id": None if pd.isna(r["batch_id"]) else str(r["batch_id"]),
                "stock_left": stock_left
            })

    if "expiry_date" in stock_df.columns and not stock_df["expiry_date"].isna().all():
        for _, r in stock_df.dropna(subset=["expiry_date"]).iterrows():
            try:
                exp = pd.to_datetime(r["expiry_date"]).date()
                days_left = (exp - today).days
                if days_left <= expiry_days:
                    alerts.append({
                        "type":"expiry_soon",
                        "product": r["product"],
                        "batch_id": None if pd.isna(r["batch_id"]) else str(r["batch_id"]),
                        "days_left": int(days_left)
                    })
            except Exception:
                pass

    return jsonify({"ok": True, "alerts": alerts, "params": {"low_threshold":low_threshold, "expiry_days":expiry_days}})

@app.route("/api/forecast")
def forecast():
    product = request.args.get("product")
    horizon = int(request.args.get("horizon", 7))
    if not product:
        return jsonify({"ok": False, "error":"Missing 'product' query param"}), 400

    with get_conn() as conn:
        df = pd.read_sql_query("SELECT product, quantity, date FROM sales WHERE product = ?", conn, params=(product,))

    if df.empty:
        return jsonify({"ok": True, "product": product, "horizon_days": horizon, "history": [], "forecast": []})

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    daily = df.groupby("date", as_index=False)["quantity"].sum().rename(columns={"date":"ds","quantity":"y"})

    history_out = [{"date": str(d.date()), "sales": int(y)} for d, y in zip(daily["ds"], daily["y"])]

    forecast_out = []

    if HAS_PROPHET:
        m = Prophet()
        m.fit(daily)
        future = m.make_future_dataframe(periods=horizon, freq="D", include_history=False)
        fcst = m.predict(future)
        for _, row in fcst.iterrows():
            forecast_out.append({
                "date": str(row["ds"].date()),
                "pred": float(row["yhat"]),
                "lower": float(row["yhat_lower"]),
                "upper": float(row["yhat_upper"])
            })
    else:
        daily_sorted = daily.sort_values("ds")
        window = min(7, len(daily_sorted))
        avg = daily_sorted["y"].tail(window).mean() if window > 0 else 0
        last_date = daily_sorted["ds"].max().date()
        for i in range(1, horizon+1):
            day = last_date + timedelta(days=i)
            forecast_out.append({
                "date": str(day),
                "pred": float(avg),
                "lower": float(max(0, avg - avg*0.2)),
                "upper": float(avg + avg*0.2)
            })

    return jsonify({"ok": True, "product": product, "horizon_days": horizon, "history": history_out, "forecast": forecast_out})

@app.route("/api/insights")
def insights():
    try:
        top_k = int(request.args.get("top_k", 5))
    except Exception:
        top_k = 5

    with get_conn() as conn:
        sales = pd.read_sql_query("SELECT product, quantity, date FROM sales", conn)

    if sales.empty:
        return jsonify({"ok": True, "daily": [], "weekly": [], "monthly": []})

    sales["date"] = pd.to_datetime(sales["date"], errors="coerce").dt.date

    today = datetime.utcnow().date()
    cutoff_daily = today - timedelta(days=1)
    cutoff_weekly = today - timedelta(days=7)
    cutoff_monthly = today - timedelta(days=30)

    def agg_sales(cutoff):
        recent = sales[sales["date"] >= cutoff]
        if recent.empty:
            return {"best": [], "least": []}
        agg = recent.groupby("product", as_index=False)["quantity"].sum().rename(columns={"quantity":"units"})
        best = agg.sort_values("units", ascending=False).head(top_k).to_dict(orient="records")
        least = agg.sort_values("units", ascending=True).head(top_k).to_dict(orient="records")
        return {"best": best, "least": least}

    return jsonify({
        "ok": True,
        "daily": agg_sales(cutoff_daily),
        "weekly": agg_sales(cutoff_weekly),
        "monthly": agg_sales(cutoff_monthly)
    })

@app.route("/api/products")
def products():
    with get_conn() as conn:
        inv = pd.read_sql_query("SELECT DISTINCT product FROM inventory", conn)
    return jsonify({"ok": True, "products": inv["product"].dropna().astype(str).tolist()})

@app.route("/api/protected")
def protected():
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        return jsonify({"ok": False, "error": "Missing or invalid token"}), 401

    token = auth_header.split("Bearer ")[1]
    # Simple token check for demo
    if token.startswith("demo-token-"):
        user_id = token.replace("demo-token-", "")
        return jsonify({"ok": True, "message": f"Welcome, {user_id}!"})

    return jsonify({"ok": False, "error": "Invalid token"}), 401

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
