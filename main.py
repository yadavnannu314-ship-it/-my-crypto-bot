"""
Binance Spot Testnet Trading Bot
=================================

Strategy (4-level confirmation):
    1. Trend filter   -> EMA50 > EMA200                 (bullish regime)
    2. Momentum band  -> RSI(14) between 40 and 58       (pullback, not overbought)
    3. Trigger        -> MACD line crosses above signal  (fresh bullish cross)
    4. Risk layer     -> ATR(14) based dynamic SL / TP

Timeframe: 15m
Trade size: 1000 USDT notional per trade (spot, long-only)

Deployment notes (Render.com):
    - Render's free/web-service tiers expect the process to bind to $PORT,
      or it will consider the deploy "unhealthy" and cycle it.
    - We run a tiny Flask app in the MAIN thread (so the port stays bound)
      and run the actual trading loop in a daemon background thread.
    - Set the Render service type to "Web Service" and start command to:
          python binance_testnet_bot.py

Environment variables required:
    BINANCE_API_KEY
    BINANCE_API_SECRET

Optional environment variables:
    SYMBOL              (default: BTCUSDT)
    INTERVAL            (default: 15m)
    TRADE_USDT          (default: 1000)
    POLL_SECONDS        (default: 60)   -> how often the loop checks for a new closed candle
    ATR_SL_MULT         (default: 1.5)
    ATR_TP_MULT         (default: 3.0)
    USE_TESTNET         (default: true)
    HTTP_PROXY / HTTPS_PROXY  -> see "Location restriction (-1009)" note below.

Install:
    pip install python-binance pandas ta flask

--------------------------------------------------------------------------
IMPORTANT - Binance APIError -1009 / geo-restriction:
--------------------------------------------------------------------------
Binance blocks API access (including Testnet) from certain regions/IP
ranges. If you deploy on a Render.com data-center IP that Binance has
blocked, you'll see something like:

    binance.exceptions.BinanceAPIException: APIError(code=-1009):
    ... restricted location ...

There is no "flag" you can send in a request that makes Binance ignore
this -- it's an IP-based block enforced at their edge. The only real
fixes are:

    1. Route your outbound traffic through an HTTP(S)/SOCKS5 proxy or a
       small VPS located in a region Binance supports, and point
       python-binance at it via `requests_params` (shown below), OR
    2. Use a Render region that isn't blocked (Binance's block list
       changes over time, so this is not guaranteed to be stable), OR
    3. Run the bot on your own VPS in a supported region and only use
       Render for the always-on Flask health-check endpoint if you want
       to keep that split.

This script reads PROXY_URL (e.g. "http://user:pass@host:port" or
"socks5h://host:port") from the environment and, if set, wires it into
both python-binance's requests session and the client's `requests_params`.
--------------------------------------------------------------------------
"""

import os
import time
import threading
import logging
from datetime import datetime, timezone

import pandas as pd
from flask import Flask, jsonify

from binance.client import Client
from binance.exceptions import BinanceAPIException, BinanceOrderException
from binance.enums import SIDE_BUY, SIDE_SELL, ORDER_TYPE_MARKET

from ta.trend import EMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

API_KEY = os.environ.get("BINANCE_API_KEY")
API_SECRET = os.environ.get("BINANCE_API_SECRET")

SYMBOL = os.environ.get("SYMBOL", "BTCUSDT")
INTERVAL = os.environ.get("INTERVAL", Client.KLINE_INTERVAL_15MINUTE if hasattr(Client, "KLINE_INTERVAL_15MINUTE") else "15m")
TRADE_USDT = float(os.environ.get("TRADE_USDT", "1000"))
POLL_SECONDS = int(os.environ.get("POLL_SECONDS", "60"))
ATR_SL_MULT = float(os.environ.get("ATR_SL_MULT", "1.5"))
ATR_TP_MULT = float(os.environ.get("ATR_TP_MULT", "3.0"))
USE_TESTNET = os.environ.get("USE_TESTNET", "true").lower() in ("1", "true", "yes")

# RSI band for level 2
RSI_LOW, RSI_HIGH = 40.0, 58.0

EMA_FAST_LEN = 50
EMA_SLOW_LEN = 200
RSI_LEN = 14
MACD_FAST, MACD_SLOW, MACD_SIGNAL = 12, 26, 9
ATR_LEN = 14

KLINES_LOOKBACK = max(EMA_SLOW_LEN * 2, 500)  # enough history for EMA200 to be meaningful

PROXY_URL = os.environ.get("PROXY_URL")  # e.g. "http://user:pass@host:port"

# --------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("binance-testnet-bot")

if not API_KEY or not API_SECRET:
    log.warning(
        "BINANCE_API_KEY / BINANCE_API_SECRET are not set. "
        "The bot will fail to authenticate until these are provided."
    )

# --------------------------------------------------------------------------
# Binance client setup (with optional proxy for -1009 geo-restriction)
# --------------------------------------------------------------------------


def build_client() -> Client:
    """
    Build a python-binance Client pointed at Testnet, optionally routed
    through a proxy to work around APIError -1009 (restricted location).
    """
    requests_params = {}

    if PROXY_URL:
        # This is the "custom request parameter" hook mentioned above:
        # python-binance forwards `requests_params` straight into every
        # underlying `requests` call, so `proxies` works exactly as it
        # would with plain `requests`.
        requests_params["proxies"] = {
            "http": PROXY_URL,
            "https": PROXY_URL,
        }
        # Give the proxy a bit more slack than the default timeout.
        requests_params["timeout"] = 20
        log.info("Routing Binance API traffic through configured proxy.")

    client = Client(
        api_key=API_KEY,
        api_secret=API_SECRET,
        testnet=USE_TESTNET,
        requests_params=requests_params or None,
    )

    # Some python-binance versions need the testnet base URLs set manually
    # for both the REST client and the underlying session.
    if USE_TESTNET:
        client.API_URL = "https://testnet.binance.vision/api"

    return client


client = build_client()

# --------------------------------------------------------------------------
# Strategy state (per-symbol, in-memory -- fine for a single-symbol bot)
# --------------------------------------------------------------------------

state_lock = threading.Lock()
bot_state = {
    "in_position": False,
    "entry_price": None,
    "qty": None,
    "stop_loss": None,
    "take_profit": None,
    "last_signal_time": None,
    "last_error": None,
    "last_checked_candle_close": None,
}

# --------------------------------------------------------------------------
# Data fetching + indicators
# --------------------------------------------------------------------------


def fetch_klines(symbol: str, interval: str, limit: int) -> pd.DataFrame:
    raw = client.get_klines(symbol=symbol, interval=interval, limit=limit)
    cols = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "num_trades",
        "taker_buy_base", "taker_buy_quote", "ignore",
    ]
    df = pd.DataFrame(raw, columns=cols)
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = df[c].astype(float)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    return df


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ema_fast"] = EMAIndicator(df["close"], window=EMA_FAST_LEN).ema_indicator()
    df["ema_slow"] = EMAIndicator(df["close"], window=EMA_SLOW_LEN).ema_indicator()
    df["rsi"] = RSIIndicator(df["close"], window=RSI_LEN).rsi()

    macd = MACD(
        df["close"],
        window_fast=MACD_FAST,
        window_slow=MACD_SLOW,
        window_sign=MACD_SIGNAL,
    )
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_hist"] = macd.macd_diff()

    atr = AverageTrueRange(df["high"], df["low"], df["close"], window=ATR_LEN)
    df["atr"] = atr.average_true_range()

    return df


def check_entry_signal(df: pd.DataFrame) -> bool:
    """
    Evaluate the 3 signal levels on the LAST CLOSED candle:
        1. EMA50 > EMA200
        2. RSI in [40, 58]
        3. MACD bullish crossover (macd crosses above signal on this candle)
    (Level 4, ATR-based SL/TP, is applied at order placement time.)
    """
    if len(df) < max(EMA_SLOW_LEN, MACD_SLOW) + 5:
        return False

    last = df.iloc[-1]
    prev = df.iloc[-2]

    trend_ok = last["ema_fast"] > last["ema_slow"]
    rsi_ok = RSI_LOW <= last["rsi"] <= RSI_HIGH
    macd_cross_up = (prev["macd"] <= prev["macd_signal"]) and (last["macd"] > last["macd_signal"])

    log.info(
        "Signal check | trend_ok=%s rsi_ok=%s (rsi=%.2f) macd_cross_up=%s "
        "(macd=%.4f sig=%.4f)",
        trend_ok, rsi_ok, last["rsi"], macd_cross_up, last["macd"], last["macd_signal"],
    )

    return bool(trend_ok and rsi_ok and macd_cross_up)


def compute_sl_tp(entry_price: float, atr_value: float) -> tuple:
    stop_loss = entry_price - ATR_SL_MULT * atr_value
    take_profit = entry_price + ATR_TP_MULT * atr_value
    return stop_loss, take_profit


# --------------------------------------------------------------------------
# Order helpers
# --------------------------------------------------------------------------


def get_symbol_filters(symbol: str) -> dict:
    info = client.get_symbol_info(symbol)
    filters = {f["filterType"]: f for f in info["filters"]}
    return {
        "step_size": float(filters["LOT_SIZE"]["stepSize"]),
        "min_qty": float(filters["LOT_SIZE"]["minQty"]),
        "tick_size": float(filters["PRICE_FILTER"]["tickSize"]),
        "min_notional": float(
            filters.get("MIN_NOTIONAL", filters.get("NOTIONAL", {})).get("minNotional", 0)
        ),
    }


def round_step(value: float, step: float) -> float:
    if step == 0:
        return value
    precision = max(0, len(str(step).split(".")[-1].rstrip("0")))
    return float(f"{(value // step) * step:.{precision}f}")


def place_market_buy(symbol: str, usdt_amount: float) -> dict:
    price = float(client.get_symbol_ticker(symbol=symbol)["price"])
    filters = get_symbol_filters(symbol)

    raw_qty = usdt_amount / price
    qty = round_step(raw_qty, filters["step_size"])

    if qty < filters["min_qty"] or qty * price < filters["min_notional"]:
        raise ValueError(
            f"Computed quantity {qty} does not meet exchange minimums "
            f"(min_qty={filters['min_qty']}, min_notional={filters['min_notional']})."
        )

    order = client.create_order(
        symbol=symbol,
        side=SIDE_BUY,
        type=ORDER_TYPE_MARKET,
        quantity=qty,
    )
    log.info("BUY order filled: qty=%s at ~%.2f", qty, price)
    return order, qty, price


def place_market_sell(symbol: str, qty: float) -> dict:
    filters = get_symbol_filters(symbol)
    qty = round_step(qty, filters["step_size"])
    order = client.create_order(
        symbol=symbol,
        side=SIDE_SELL,
        type=ORDER_TYPE_MARKET,
        quantity=qty,
    )
    log.info("SELL order filled: qty=%s", qty)
    return order


# --------------------------------------------------------------------------
# Trading loop (runs in a background thread)
# --------------------------------------------------------------------------


def manage_open_position(current_price: float):
    with state_lock:
        if not bot_state["in_position"]:
            return
        sl = bot_state["stop_loss"]
        tp = bot_state["take_profit"]
        qty = bot_state["qty"]

    if sl is None or tp is None or qty is None:
        return

    if current_price <= sl or current_price >= tp:
        reason = "stop-loss" if current_price <= sl else "take-profit"
        try:
            place_market_sell(SYMBOL, qty)
            log.info("Position closed via %s at price %.2f", reason, current_price)
        except (BinanceAPIException, BinanceOrderException, ValueError) as e:
            log.error("Failed to close position (%s): %s", reason, e)
            with state_lock:
                bot_state["last_error"] = str(e)
            return

        with state_lock:
            bot_state.update(
                in_position=False,
                entry_price=None,
                qty=None,
                stop_loss=None,
                take_profit=None,
            )


def try_enter_position(df: pd.DataFrame):
    with state_lock:
        if bot_state["in_position"]:
            return

    if not check_entry_signal(df):
        return

    last = df.iloc[-1]
    atr_value = last["atr"]

    if pd.isna(atr_value) or atr_value <= 0:
        log.warning("ATR unavailable/invalid, skipping entry this cycle.")
        return

    try:
        order, qty, fill_price = place_market_buy(SYMBOL, TRADE_USDT)
    except (BinanceAPIException, BinanceOrderException, ValueError) as e:
        log.error("Failed to enter position: %s", e)
        with state_lock:
            bot_state["last_error"] = str(e)
        return

    stop_loss, take_profit = compute_sl_tp(fill_price, atr_value)

    with state_lock:
        bot_state.update(
            in_position=True,
            entry_price=fill_price,
            qty=qty,
            stop_loss=stop_loss,
            take_profit=take_profit,
            last_signal_time=datetime.now(timezone.utc).isoformat(),
        )

    log.info(
        "Entered LONG %s qty=%s entry=%.2f SL=%.2f TP=%.2f (ATR=%.4f)",
        SYMBOL, qty, fill_price, stop_loss, take_profit, atr_value,
    )


def trading_loop():
    log.info(
        "Starting trading loop | symbol=%s interval=%s trade_usdt=%s testnet=%s",
        SYMBOL, INTERVAL, TRADE_USDT, USE_TESTNET,
    )

    while True:
        try:
            df = fetch_klines(SYMBOL, INTERVAL, KLINES_LOOKBACK)
            df = add_indicators(df)

            last_closed = df.iloc[-1]
            last_close_time = last_closed["close_time"]

            current_price = float(client.get_symbol_ticker(symbol=SYMBOL)["price"])

            # Always manage an existing position on every poll (SL/TP can
            # trigger intra-candle, not just on candle close).
            manage_open_position(current_price)

            # Only look for NEW entries once per newly closed candle.
            with state_lock:
                already_checked = bot_state["last_checked_candle_close"]

            if already_checked != last_close_time:
                try_enter_position(df)
                with state_lock:
                    bot_state["last_checked_candle_close"] = last_close_time

            with state_lock:
                bot_state["last_error"] = None

        except BinanceAPIException as e:
            # Explicitly surface the -1009 geo-restriction case with guidance.
            if getattr(e, "code", None) == -1009:
                log.error(
                    "APIError -1009 (restricted location) from Binance. "
                    "Set PROXY_URL to a working proxy/VPS in a supported "
                    "region, or move the deployment. See module docstring."
                )
            else:
                log.error("Binance API error: %s", e)
            with state_lock:
                bot_state["last_error"] = str(e)

        except Exception as e:  # noqa: BLE001 - keep the loop alive no matter what
            log.exception("Unexpected error in trading loop: %s", e)
            with state_lock:
                bot_state["last_error"] = str(e)

        time.sleep(POLL_SECONDS)


# --------------------------------------------------------------------------
# Flask app (main thread) - keeps Render's port-binding health check happy
# --------------------------------------------------------------------------

app = Flask(__name__)


@app.route("/")
def index():
    return jsonify({"status": "ok", "message": "Binance testnet bot is running."})


@app.route("/status")
def status():
    with state_lock:
        return jsonify(
            {
                "symbol": SYMBOL,
                "interval": INTERVAL,
                "trade_usdt": TRADE_USDT,
                "testnet": USE_TESTNET,
                **bot_state,
                "last_checked_candle_close": (
                    str(bot_state["last_checked_candle_close"])
                    if bot_state["last_checked_candle_close"] is not None
                    else None
                ),
            }
        )


@app.route("/health")
def health():
    # Simple liveness probe for Render.
    return jsonify({"status": "healthy"}), 200


def start_background_trading_thread():
    thread = threading.Thread(target=trading_loop, daemon=True)
    thread.start()
    return thread


if __name__ == "__main__":
    start_background_trading_thread()

    port = int(os.environ.get("PORT", "10000"))
    # Flask runs in the MAIN thread; the trading loop runs in the daemon
    # background thread started above. This is what keeps Render happy
    # (a bound port) while trading continues independently.
    app.run(host="0.0.0.0", port=port)
