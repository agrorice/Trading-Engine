import pandas as pd
import numpy as np

# --- ACCOUNT CONFIG ---
ACCOUNT_SIZE  = 5000.00   # Starting account in dollars
RISK_PERCENT  = 3.0       # % of account risked per trade (adjust freely)

# Derived — do not edit this line
RISK_DOLLARS  = ACCOUNT_SIZE * (RISK_PERCENT / 100)   # e.g. 3% of $5000 = $150

# --- SESSION TIMES (all in New York / US Eastern time) ---
LONDON_SESSION_START = 3    # 3:00 AM NY
LONDON_SESSION_END   = 8    # 8:00 AM NY
NY_SESSION_START     = 8    # 8:00 AM NY
NY_SESSION_END       = 13   # 1:00 PM NY


def backtest(df_1min):
    """
    df_1min: 1-minute OHLC dataframe with column 'datetime' as string/datetime
    Returns: list of trade result dicts

    Strategy:
      1. London session (3am-8am NY) establishes the session high/low.
      2. NY session (8am-1pm NY) scans 5min candles for:
           - Sweep of London high or low
           - Break of Structure (BOS)
           - FVG or IFVG confirmation
      3. Full TP = London session high (long) or London session low (short)
      4. At 50% of price distance to full TP: close Half 1, move stop to breakeven
      5. Half 2 then runs to full TP or exits at breakeven
      6. One trade per day maximum
    """

    # --- Localise to New York time ---
    df_1min['datetime'] = pd.to_datetime(df_1min['datetime'])

    if df_1min['datetime'].dt.tz is None:
        df_1min['datetime'] = (
            df_1min['datetime']
            .dt.tz_localize('UTC')
            .dt.tz_convert('America/New_York')
        )
    else:
        df_1min['datetime'] = df_1min['datetime'].dt.tz_convert('America/New_York')

    df_1min.set_index('datetime', inplace=True)

    # Resample to 5-minute candles
    df_5min = df_1min.resample('5min').agg({
        'open':  'first',
        'high':  'max',
        'low':   'min',
        'close': 'last'
    }).dropna().reset_index()

    trades = []
    last_trade_day = None

    # --- Pre-compute London session high/low per calendar day ---
    london_mask = (
        (df_5min['datetime'].dt.hour >= LONDON_SESSION_START) &
        (df_5min['datetime'].dt.hour <  LONDON_SESSION_END)
    )
    london_candles = df_5min[london_mask].copy()
    london_candles['date'] = london_candles['datetime'].dt.date

    london_levels = (
        london_candles
        .groupby('date')
        .agg(london_high=('high', 'max'), london_low=('low', 'min'))
        .reset_index()
    )
    london_map = {
        row['date']: (row['london_high'], row['london_low'])
        for _, row in london_levels.iterrows()
    }

    # --- Scan NY session candles for setups ---
    ny_mask = (
        (df_5min['datetime'].dt.hour >= NY_SESSION_START) &
        (df_5min['datetime'].dt.hour <  NY_SESSION_END)
    )
    df_ny = df_5min[ny_mask].reset_index(drop=True)

    for i in range(3, len(df_ny)):
        current_day = df_ny['datetime'][i].date()

        if last_trade_day == current_day:
            continue

        if current_day not in london_map:
            continue

        london_high, london_low = london_map[current_day]

        # --- SWEEP of London session high or low ---
        swept_high = any(df_ny['high'][i - k] > london_high for k in range(1, 4))
        swept_low  = any(df_ny['low'][i - k]  < london_low  for k in range(1, 4))
        sweep = swept_high or swept_low

        if not sweep:
            continue

        # --- BOS (Break of Structure) ---
        bos_bull = df_ny['close'][i] > df_ny['high'][i-1]
        bos_bear = df_ny['close'][i] < df_ny['low'][i-1]
        bos = bos_bull or bos_bear

        if not bos:
            continue

        # --- FVG / IFVG ---
        fvg_up   = df_ny['low'][i]  > df_ny['high'][i-2]
        fvg_down = df_ny['high'][i] < df_ny['low'][i-2]
        fvg = fvg_up or fvg_down

        ifvg_up   = df_ny['low'][i]  > df_ny['low'][i-1]  and df_ny['high'][i] < df_ny['high'][i-1]
        ifvg_down = df_ny['high'][i] < df_ny['high'][i-1] and df_ny['low'][i]  > df_ny['low'][i-1]
        ifvg = ifvg_up or ifvg_down

        if not (fvg or ifvg):
            continue

        # --- ENTRY ---
        entry   = df_ny['close'][i]
        is_long = bool(bos_bull)   # direction follows the BOS candle

        stop    = df_ny['low'][i]  if is_long else df_ny['high'][i]
        full_tp = london_high      if is_long else london_low

        # Sanity: TP must be beyond entry in the trade direction
        if is_long  and full_tp <= entry:
            continue
        if not is_long and full_tp >= entry:
            continue

        risk_distance = abs(entry - stop)
        if risk_distance == 0:
            continue

        # 50% partial TP — midpoint between entry and full TP
        half_tp = entry + (full_tp - entry) * 0.5

        # --- SIMULATE two-half-lot trade ---
        outcome = simulate_trade_split(df_ny, i, entry, stop, half_tp, full_tp, is_long)

        # Scale raw price-distance profit to dollars
        # RISK_DOLLARS covers the full risk distance on the full lot
        dollar_per_pip  = RISK_DOLLARS / risk_distance
        profit_dollars  = round(outcome['profit_raw'] * dollar_per_pip, 2)
        profit_percent  = round((profit_dollars / ACCOUNT_SIZE) * 100, 2)

        rr = round(abs(full_tp - entry) / risk_distance, 2)

        trades.append({
            'date':           current_day,
            'direction':      'LONG' if is_long else 'SHORT',
            'entry':          round(entry, 5),
            'stop':           round(stop, 5),
            'half_tp':        round(half_tp, 5),
            'full_tp':        round(full_tp, 5),
            'R:R':            f"1:{rr} ({RISK_PERCENT:.1f}% risk / ${RISK_DOLLARS:.0f})",
            'profit_percent': profit_percent,
            'profit_dollars': profit_dollars,
            'W/L':            outcome['label'],   # 'W', 'PW', 'L', 'BE'
        })

        last_trade_day = current_day

    return trades


def simulate_trade_split(df, entry_idx, entry, stop, half_tp, full_tp, is_long):
    """
    Simulates a split-lot trade candle by candle after the entry candle.

    Two halves:
      Half 1 — closes at half_tp (50% of TP distance). Profit locked in.
               Stop on Half 2 then moves to breakeven (entry).
      Half 2 — runs from entry to full_tp, or exits at breakeven if Half 1 hit.

    Returns dict:
      profit_raw  — combined price-distance profit across both halves
                    (caller multiplies by dollar_per_pip to get $)
      label       — 'W'  : both halves hit TP
                    'PW' : Half 1 hit, Half 2 exited at breakeven (or session end)
                    'L'  : stopped out before Half 1 hit
                    'BE' : session ended without any TP hit and no stop
    """
    half1_done   = False
    half1_profit = 0.0

    for j in range(entry_idx + 1, len(df)):
        candle_low  = df['low'][j]
        candle_high = df['high'][j]

        # Active stop: breakeven once Half 1 is closed, original stop otherwise
        active_stop = entry if half1_done else stop

        if is_long:
            # --- Stop check ---
            if candle_low <= active_stop:
                half2_profit = active_stop - entry   # 0 at BE, negative at full stop
                return {
                    'profit_raw': half1_profit + half2_profit,
                    'label': 'PW' if half1_done else 'L'
                }

            # --- Half 1 TP check ---
            if not half1_done and candle_high >= half_tp:
                half1_profit = half_tp - entry        # positive price gain
                half1_done   = True

            # --- Full TP check (Half 2) ---
            if half1_done and candle_high >= full_tp:
                half2_profit = full_tp - entry
                return {
                    'profit_raw': half1_profit + half2_profit,
                    'label': 'W'
                }

        else:  # SHORT
            # --- Stop check ---
            if candle_high >= active_stop:
                half2_profit = entry - active_stop    # 0 at BE, negative at full stop
                return {
                    'profit_raw': half1_profit + half2_profit,
                    'label': 'PW' if half1_done else 'L'
                }

            # --- Half 1 TP check ---
            if not half1_done and candle_low <= half_tp:
                half1_profit = entry - half_tp
                half1_done   = True

            # --- Full TP check (Half 2) ---
            if half1_done and candle_low <= full_tp:
                half2_profit = entry - full_tp
                return {
                    'profit_raw': half1_profit + half2_profit,
                    'label': 'W'
                }

    # Session ended without resolution
    if half1_done:
        # Half 1 locked in, Half 2 still open — count as partial win
        return {'profit_raw': half1_profit, 'label': 'PW'}

    return {'profit_raw': 0.0, 'label': 'BE'}