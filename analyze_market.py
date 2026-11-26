#!/usr/bin/env python3
"""
analyze_market.py - simple technical analysis for any stock / index / gold / crypto.

Input : a CSV of daily (or any timeframe) candles with columns
        date/time, open, high, low, close, volume
        (works with TradingView exports, Yahoo Finance and NSE downloads).
Output: a text report with trend, momentum, volatility, support/resistance
        and an overall bias. Optionally saves a chart.

Usage
-----
    python analyze_market.py nifty.csv
    python analyze_market.py nifty.csv --chart          # also saves nifty_chart.png
    python analyze_market.py --fetch ^NSEI --chart      # download with yfinance (if installed)

Only the Python standard library is required. matplotlib is used for --chart,
yfinance for --fetch; both are optional.

This is an educational tool, not financial advice. Indicators describe the
past; they do not predict the future.
"""

import argparse
import csv
import math
import os
import sys
from datetime import datetime


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------

def _num(text):
    if text is None:
        return None
    text = str(text).strip().replace(",", "")
    if text in ("", "-", "null", "None", "nan", "NaN"):
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _parse_date(text):
    text = str(text).strip()
    if text.isdigit():  # unix timestamp (TradingView exports use this)
        try:
            return datetime.fromtimestamp(int(text))
        except (ValueError, OSError):
            return None
    for fmt in ("%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%d-%m-%Y", "%d-%b-%Y",
                "%d/%m/%Y", "%m/%d/%Y", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(text[:len(datetime.now().strftime(fmt))], fmt)
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")[:19])
    except ValueError:
        return None


def load_csv(path):
    with open(path, newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        if not reader.fieldnames:
            sys.exit("The CSV file looks empty.")
        cols = {name.strip().lower(): name for name in reader.fieldnames}

        def pick(*names):
            for n in names:
                if n in cols:
                    return cols[n]
            return None

        c_date = pick("date", "time", "datetime", "timestamp")
        c_open = pick("open")
        c_high = pick("high")
        c_low = pick("low")
        c_close = pick("close", "close price", "adj close")
        c_vol = pick("volume", "vol", "shares traded")
        if not c_close:
            sys.exit("Could not find a 'close' column. Found: %s" % list(cols))

        rows = []
        for r in reader:
            close = _num(r.get(c_close))
            if close is None:
                continue
            rows.append({
                "date": _parse_date(r[c_date]) if c_date else None,
                "open": _num(r.get(c_open)) if c_open else None,
                "high": _num(r.get(c_high)) if c_high else None,
                "low": _num(r.get(c_low)) if c_low else None,
                "close": close,
                "volume": _num(r.get(c_vol)) if c_vol else None,
            })

    if all(r["date"] for r in rows):
        rows.sort(key=lambda r: r["date"])
    for r in rows:  # fill missing OHLC with close so every indicator works
        for k in ("open", "high", "low"):
            if r[k] is None:
                r[k] = r["close"]
    return rows


def fetch_yfinance(symbol, period="2y"):
    try:
        import yfinance as yf
    except ImportError:
        sys.exit("yfinance is not installed. Run: pip install yfinance  "
                 "(or export a CSV from TradingView and pass the file instead).")
    df = yf.Ticker(symbol).history(period=period, auto_adjust=False)
    if df.empty:
        sys.exit("No data returned for %s" % symbol)
    rows = []
    for idx, r in df.iterrows():
        rows.append({
            "date": idx.to_pydatetime().replace(tzinfo=None),
            "open": float(r["Open"]), "high": float(r["High"]),
            "low": float(r["Low"]), "close": float(r["Close"]),
            "volume": float(r["Volume"]) if r["Volume"] == r["Volume"] else None,
        })
    return rows


# --------------------------------------------------------------------------
# Indicators (lists are padded with None where there is not enough history)
# --------------------------------------------------------------------------

def sma(values, n):
    out = [None] * len(values)
    window_sum = 0.0
    for i, v in enumerate(values):
        window_sum += v
        if i >= n:
            window_sum -= values[i - n]
        if i >= n - 1:
            out[i] = window_sum / n
    return out


def ema(values, n):
    out = [None] * len(values)
    if len(values) < n:
        return out
    k = 2.0 / (n + 1)
    prev = sum(values[:n]) / n
    out[n - 1] = prev
    for i in range(n, len(values)):
        prev = values[i] * k + prev * (1 - k)
        out[i] = prev
    return out


def rsi(values, n=14):
    out = [None] * len(values)
    if len(values) <= n:
        return out
    gains = losses = 0.0
    for i in range(1, n + 1):
        d = values[i] - values[i - 1]
        gains += max(d, 0)
        losses += max(-d, 0)
    avg_g, avg_l = gains / n, losses / n
    out[n] = 100.0 if avg_l == 0 else 100 - 100 / (1 + avg_g / avg_l)
    for i in range(n + 1, len(values)):
        d = values[i] - values[i - 1]
        avg_g = (avg_g * (n - 1) + max(d, 0)) / n
        avg_l = (avg_l * (n - 1) + max(-d, 0)) / n
        out[i] = 100.0 if avg_l == 0 else 100 - 100 / (1 + avg_g / avg_l)
    return out


def macd(values, fast=12, slow=26, signal=9):
    ef, es = ema(values, fast), ema(values, slow)
    line = [None if a is None or b is None else a - b for a, b in zip(ef, es)]
    first = next((i for i, v in enumerate(line) if v is not None), None)
    sig = [None] * len(values)
    if first is not None:
        sub = ema([v for v in line[first:]], signal)
        for j, v in enumerate(sub):
            sig[first + j] = v
    hist = [None if a is None or b is None else a - b for a, b in zip(line, sig)]
    return line, sig, hist


def bollinger(values, n=20, width=2.0):
    mid = sma(values, n)
    up, lo = [None] * len(values), [None] * len(values)
    for i in range(n - 1, len(values)):
        win = values[i - n + 1:i + 1]
        sd = math.sqrt(sum((x - mid[i]) ** 2 for x in win) / n)
        up[i], lo[i] = mid[i] + width * sd, mid[i] - width * sd
    return up, mid, lo


def atr(rows, n=14):
    out = [None] * len(rows)
    if len(rows) <= n:
        return out
    trs = [None]
    for i in range(1, len(rows)):
        h, l, pc = rows[i]["high"], rows[i]["low"], rows[i - 1]["close"]
        trs.append(max(h - l, abs(h - pc), abs(l - pc)))
    prev = sum(trs[1:n + 1]) / n
    out[n] = prev
    for i in range(n + 1, len(rows)):
        prev = (prev * (n - 1) + trs[i]) / n
        out[i] = prev
    return out


# --------------------------------------------------------------------------
# Analysis
# --------------------------------------------------------------------------

def pct(a, b):
    return (a / b - 1) * 100 if b else 0.0


def analyze(rows):
    closes = [r["close"] for r in rows]
    n = len(closes)
    if n < 30:
        sys.exit("Need at least 30 candles for a meaningful analysis (got %d)." % n)

    ind = {
        "sma20": sma(closes, 20), "sma50": sma(closes, 50), "sma200": sma(closes, 200),
        "rsi": rsi(closes), "atr": atr(rows),
    }
    ind["macd"], ind["macd_sig"], ind["macd_hist"] = macd(closes)
    ind["bb_up"], ind["bb_mid"], ind["bb_lo"] = bollinger(closes)

    last = n - 1
    g = lambda key: ind[key][last]
    price = closes[last]
    notes, score = [], 0

    # ---- trend
    s20, s50, s200 = g("sma20"), g("sma50"), g("sma200")
    for label, val in (("20-day", s20), ("50-day", s50), ("200-day", s200)):
        if val is None:
            continue
        above = price > val
        notes.append("Price is %s the %s average (%.2f), %.1f%% %s." % (
            "above" if above else "below", label, val, abs(pct(price, val)),
            "above" if above else "below"))
        if label != "20-day":
            score += 1 if above else -1
    if s50 is not None and s200 is not None:
        if s50 > s200:
            notes.append("50-day average is above the 200-day average (long-term uptrend).")
            score += 1
        else:
            notes.append("50-day average is below the 200-day average (long-term downtrend).")
            score -= 1
        # recent crossover within the last 10 candles
        for i in range(max(last - 10, 1), last + 1):
            a0, b0, a1, b1 = ind["sma50"][i - 1], ind["sma200"][i - 1], ind["sma50"][i], ind["sma200"][i]
            if None in (a0, b0, a1, b1):
                continue
            if a0 <= b0 and a1 > b1:
                notes.append("Golden cross (50 over 200) happened in the last 10 candles.")
            elif a0 >= b0 and a1 < b1:
                notes.append("Death cross (50 under 200) happened in the last 10 candles.")

    # ---- momentum
    r = g("rsi")
    if r is not None:
        if r >= 70:
            notes.append("RSI(14) is %.1f: overbought zone, a pullback becomes more likely." % r)
        elif r <= 30:
            notes.append("RSI(14) is %.1f: oversold zone, a bounce becomes more likely." % r)
        elif r >= 50:
            notes.append("RSI(14) is %.1f: momentum leans positive." % r)
            score += 1
        else:
            notes.append("RSI(14) is %.1f: momentum leans negative." % r)
            score -= 1
    m, ms, mh = g("macd"), g("macd_sig"), g("macd_hist")
    if None not in (m, ms, mh):
        prev_h = ind["macd_hist"][last - 1]
        rising = prev_h is not None and mh > prev_h
        if m > ms:
            notes.append("MACD is above its signal line (bullish), histogram %s." %
                         ("rising" if rising else "fading"))
            score += 1
        else:
            notes.append("MACD is below its signal line (bearish), histogram %s." %
                         ("recovering" if rising else "falling"))
            score -= 1

    # ---- volatility
    up, lo, a = g("bb_up"), g("bb_lo"), g("atr")
    if up is not None and up != lo:
        pos = (price - lo) / (up - lo) * 100
        txt = "Price sits at %.0f%% of its Bollinger band range" % pos
        if price > up:
            txt += " (above the upper band, stretched)."
        elif price < lo:
            txt += " (below the lower band, stretched)."
        else:
            txt += "."
        notes.append(txt)
    if a is not None:
        notes.append("ATR(14) is %.2f, about %.2f%% of price: typical daily swing." % (a, a / price * 100))

    # ---- levels
    window = rows[-20:]
    support = min(x["low"] for x in window)
    resistance = max(x["high"] for x in window)
    lookback = rows[-252:]
    hi52 = max(x["high"] for x in lookback)
    lo52 = min(x["low"] for x in lookback)

    # ---- volume
    vol_note = None
    vols = [x["volume"] for x in rows if x["volume"]]
    if len(vols) >= 21 and rows[-1]["volume"]:
        avg20 = sum(vols[-21:-1]) / 20
        ratio = rows[-1]["volume"] / avg20 if avg20 else 0
        vol_note = "Latest volume is %.1fx the 20-candle average." % ratio

    # ---- returns
    def ret(k):
        return pct(price, closes[-1 - k]) if n > k else None

    if score >= 3:
        bias = "BULLISH"
    elif score <= -3:
        bias = "BEARISH"
    else:
        bias = "NEUTRAL / MIXED"

    return {
        "price": price, "prev": closes[-2], "date": rows[-1]["date"], "n": n,
        "ret": {"1": ret(1), "5": ret(5), "20": ret(20), "60": ret(60)},
        "notes": notes, "support": support, "resistance": resistance,
        "hi52": hi52, "lo52": lo52, "vol_note": vol_note,
        "score": score, "bias": bias, "ind": ind,
    }


def print_report(res, title):
    line = "=" * 64
    print(line)
    print(" MARKET ANALYSIS: %s" % title)
    print(line)
    when = res["date"].strftime("%d %b %Y") if res["date"] else "last candle"
    print(" Last close (%s): %.2f  (%+.2f%% vs previous)" %
          (when, res["price"], pct(res["price"], res["prev"])))
    parts = ["%sd %+.1f%%" % (k, v) for k, v in res["ret"].items() if v is not None]
    print(" Returns: " + " | ".join(parts))
    print(" Candles analysed: %d" % res["n"])
    print("-" * 64)
    for note in res["notes"]:
        print(" * " + note)
    if res["vol_note"]:
        print(" * " + res["vol_note"])
    print("-" * 64)
    print(" 20-candle support    : %.2f" % res["support"])
    print(" 20-candle resistance : %.2f" % res["resistance"])
    print(" 52-week range        : %.2f - %.2f (price is %.1f%% below the high)" %
          (res["lo52"], res["hi52"], abs(pct(res["price"], res["hi52"]))))
    print("-" * 64)
    print(" Indicator score: %+d   Overall bias: %s" % (res["score"], res["bias"]))
    print(" (Score ranges about -5 to +5. It summarises the past, it does not")
    print("  predict the future. Not financial advice.)")
    print(line)


def save_chart(rows, res, path, title):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed, so no chart was saved "
              "(pip install matplotlib to enable --chart).")
        return
    ind = res["ind"]
    xs = list(range(len(rows)))
    closes = [r["close"] for r in rows]
    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, figsize=(11, 9), sharex=True, gridspec_kw={"height_ratios": [3, 1, 1]})
    ax1.plot(xs, closes, label="Close", linewidth=1.4)
    for key, lab in (("sma20", "SMA20"), ("sma50", "SMA50"), ("sma200", "SMA200")):
        ax1.plot(xs, [v if v is not None else float("nan") for v in ind[key]], label=lab, linewidth=1)
    ax1.fill_between(xs,
                     [v if v is not None else float("nan") for v in ind["bb_lo"]],
                     [v if v is not None else float("nan") for v in ind["bb_up"]],
                     alpha=0.1, label="Bollinger")
    ax1.set_title("%s  |  bias: %s" % (title, res["bias"]))
    ax1.legend(loc="upper left", fontsize=8)
    ax2.plot(xs, [v if v is not None else float("nan") for v in ind["rsi"]])
    ax2.axhline(70, linestyle="--", linewidth=0.8)
    ax2.axhline(30, linestyle="--", linewidth=0.8)
    ax2.set_ylabel("RSI")
    ax3.bar(xs, [v if v is not None else 0 for v in ind["macd_hist"]])
    ax3.plot(xs, [v if v is not None else float("nan") for v in ind["macd"]], linewidth=1)
    ax3.plot(xs, [v if v is not None else float("nan") for v in ind["macd_sig"]], linewidth=1)
    ax3.set_ylabel("MACD")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print("Chart saved to %s" % path)


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Simple technical market analysis")
    ap.add_argument("csv", nargs="?", help="CSV file with OHLCV candles")
    ap.add_argument("--fetch", metavar="SYMBOL", help="download data with yfinance, e.g. ^NSEI, RELIANCE.NS, GC=F")
    ap.add_argument("--period", default="2y", help="period for --fetch (default 2y)")
    ap.add_argument("--chart", action="store_true", help="save a PNG chart")
    args = ap.parse_args()

    if args.fetch:
        rows, title = fetch_yfinance(args.fetch, args.period), args.fetch
        base = args.fetch.replace("^", "").replace("=", "_").replace(".", "_")
    elif args.csv:
        rows = load_csv(args.csv)
        title = os.path.basename(args.csv)
        base = os.path.splitext(args.csv)[0]
    else:
        ap.error("give a CSV file or use --fetch SYMBOL")

    res = analyze(rows)
    print_report(res, title)
    if args.chart:
        save_chart(rows, res, base + "_chart.png", title)


if __name__ == "__main__":
    main()
