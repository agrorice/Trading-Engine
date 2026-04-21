"""
chart_server.py — API backend for the trade chart.

HOW TO USE:
  1. Make sure your API keys are loaded:
       source /Users/andrew/.keys
  2. Run:
       python chart_server.py
  3. Open chart.html with VS Code Live Server (right-click → Open with Live Server)

The server only provides the /api/backtest endpoint — Live Server handles the HTML.
"""

import os
import datetime
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

from data_fetcher  import fetch_5min
from news_calendar import build_calendar, get_pair
from strategy      import run_day
from engine        import EURUSD_UNITS, USDJPY_UNITS

app = Flask(__name__, static_folder=os.path.dirname(os.path.abspath(__file__)))
CORS(app)


def _dt_to_unix(dt) -> int | None:
    """Convert a timezone-aware pandas Timestamp (or datetime) to UTC Unix seconds."""
    if dt is None:
        return None
    try:
        return int(pd.Timestamp(dt).timestamp())
    except Exception:
        return None


@app.route("/")
def index():
    return send_from_directory(os.path.dirname(os.path.abspath(__file__)), "chart.html")


@app.route("/api/backtest")
def api_backtest():
    start_date = request.args.get("start", "").strip()
    end_date   = request.args.get("end", "").strip()

    if not start_date or not end_date:
        return jsonify({"error": "Missing start or end date"}), 400

    try:
        datetime.date.fromisoformat(start_date)
        datetime.date.fromisoformat(end_date)
    except ValueError:
        return jsonify({"error": "Invalid date format — use YYYY-MM-DD"}), 400

    print(f"\n[chart] Backtest request: {start_date} → {end_date}")

    # ------------------------------------------------------------------
    # Build calendar + fetch price data
    # ------------------------------------------------------------------
    try:
        calendar   = build_calendar(start_date, end_date)
        eurusd_df  = fetch_5min("EURUSD", start_date, end_date)
        usdjpy_df  = fetch_5min("USDJPY", start_date, end_date)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    def split_by_day(df: pd.DataFrame) -> dict:
        df = df.copy()
        df["_date"] = df["datetime"].dt.date
        return {date: group.drop(columns="_date").reset_index(drop=True)
                for date, group in df.groupby("_date")}

    eurusd_by_day = split_by_day(eurusd_df)
    usdjpy_by_day = split_by_day(usdjpy_df)

    start_d = datetime.date.fromisoformat(start_date)
    end_d   = datetime.date.fromisoformat(end_date)
    all_dates = sorted(set(eurusd_by_day.keys()) | set(usdjpy_by_day.keys()))
    all_dates = [d for d in all_dates if start_d <= d <= end_d]

    # ------------------------------------------------------------------
    # Run strategy per day, collect candles + trade data
    # ------------------------------------------------------------------
    days = []

    for date in all_dates:
        pair = get_pair(date, calendar)
        if pair is None:
            continue  # USD bank holiday

        day_df = eurusd_by_day.get(date) if pair == "EURUSD" else usdjpy_by_day.get(date)
        if day_df is None or day_df.empty:
            continue

        # Serialize candles for the chart (Unix timestamps)
        candles_json = []
        for _, row in day_df.iterrows():
            candles_json.append({
                "time":  _dt_to_unix(row["datetime"]),
                "open":  float(row["open"]),
                "high":  float(row["high"]),
                "low":   float(row["low"]),
                "close": float(row["close"]),
            })

        result = run_day(day_df, pair)
        trade_json = None

        if result is not None and result["risk_dist"] != 0:
            outcome    = result["outcome"]
            risk_dist  = result["risk_dist"]
            rr         = round(abs(result["full_tp"] - result["entry"]) / risk_dist, 2)
            r_result   = round(outcome["profit_raw"] / risk_dist, 2)

            if pair == "EURUSD":
                profit_dollars = round(EURUSD_UNITS * outcome["profit_raw"], 2)
            else:
                profit_dollars = round(USDJPY_UNITS * outcome["profit_raw"] / result["entry"], 2)

            # Map candle indices → Unix timestamps for chart annotations
            def idx_to_unix(idx):
                if idx is None or idx >= len(day_df):
                    return None
                return _dt_to_unix(day_df.iloc[idx]["datetime"])

            trade_json = {
                "entry":          result["entry"],
                "stop":           result["stop"],
                "tp1":            result["tp1"],
                "full_tp":        result["full_tp"],
                "psh":            result["psh"],
                "psl":            result["psl"],
                "midpoint":       result["midpoint"],
                "direction":      result["direction"],
                "sweep":          result["sweep_direction"].upper(),
                "wl":             outcome["label"],
                "rr":             f"1:{rr}",
                "r_result":       r_result,
                "profit_dollars": profit_dollars,
                # Timestamps for chart annotations
                "entry_time":     idx_to_unix(result.get("entry_idx")),
                "sweep_time":     idx_to_unix(result.get("sweep_idx")),
                "bos_time":       idx_to_unix(result.get("bos_idx")),
                "tp1_time":       _dt_to_unix(outcome.get("tp1_time")),
                "exit_time":      _dt_to_unix(outcome.get("exit_time")),
            }

        days.append({
            "date":    str(date),
            "pair":    pair,
            "candles": candles_json,
            "trade":   trade_json,
        })

    print(f"[chart] Returning {len(days)} days, "
          f"{sum(1 for d in days if d['trade'] is not None)} with trades")

    return jsonify({"days": days, "start": start_date, "end": end_date})


if __name__ == "__main__":
    if not os.environ.get("TWELVE_DATA_API_KEY"):
        print("ERROR: TWELVE_DATA_API_KEY not found.")
        print("Run:  source /Users/andrew/.keys   then try again.")
        raise SystemExit(1)

    print("=" * 60)
    print("  Trade Chart Server")
    print("  Open: http://localhost:5001")
    print("=" * 60)
    app.run(debug=False, port=5001, use_reloader=False)
