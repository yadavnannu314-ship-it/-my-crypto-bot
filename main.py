import os
import time
import threading
from flask import Flask
import pandas as pd
from binance.client import Client
from ta.momentum import RSIIndicator
from ta.trend import MACD, EMAIndicator
from ta.volatility import AverageTrueRange

# 1. Flask Web Server (क्लाउड को 24 घंटे जगाए रखने के लिए)
app = Flask('')

@app.route('/')
def home():
    return "🚀 बॉट क्लाउड पर 24/7 सफलतापूर्वक चल रहा है!"

def run_web_server():
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)

# 2. आपका एडवांस 4-लेवल ट्रेडिंग बॉट लॉजिक
API_KEY = os.environ.get('BINANCE_API_KEY')
API_SECRET = os.environ.get('BINANCE_API_SECRET')
client = Client(API_KEY, API_SECRET, testnet=True)
SYMBOL = 'BTCUSDT'
QUANTITY = 0.001

def get_market_data():
    try:
        bars = client.get_historical_klines(SYMBOL, Client.KLINE_INTERVAL_1HOUR, "250 hours ago UTC")
        df = pd.DataFrame(bars, columns=['time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'asset_volume', 'trades', 'takers_buy_base', 'takers_buy_quote', 'ignored'])
        df['close'] = pd.to_numeric(df['close'])
        df['high'] = pd.to_numeric(df['high'])
        df['low'] = pd.to_numeric(df['low'])
        return df
    except Exception as e:
        print(f"⚠️ डेटा लेने में समस्या आई: {e}")
        return None

def analyze_market():
    df = get_market_data()
    if df is None or len(df) < 200:
        return 'HOLD', 0, 0

    ema_50 = EMAIndicator(close=df['close'], window=50).ema_indicator()
    ema_200 = EMAIndicator(close=df['close'], window=200).ema_indicator()
    rsi = RSIIndicator(close=df['close'], window=14).rsi()
    macd_obj = MACD(close=df['close'])
    macd_line = macd_obj.macd()
    macd_signal = macd_obj.macd_signal()
    atr_obj = AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=14)
    atr = atr_obj.average_true_range()
    
    last_idx = df.index[-1]
    prev_idx = df.index[-2]
    current_price = df['close'].loc[last_idx]
    
    print(f"\n📊 [LIVE] Price: {current_price} | RSI: {rsi.loc[last_idx]:.2f}")
    
    is_bullish = ema_50.loc[last_idx] > ema_200.loc[last_idx]
    macd_cross_up = (macd_line.loc[prev_idx] <= macd_signal.loc[prev_idx]) and (macd_line.loc[last_idx] > macd_signal.loc[last_idx])
    
    if is_bullish and (40 <= rsi.loc[last_idx] <= 58) and macd_cross_up:
        return 'BUY', current_price, atr.loc[last_idx]

    is_bearish = ema_50.loc[last_idx] < ema_200.loc[last_idx]
    macd_cross_down = (macd_line.loc[prev_idx] >= macd_signal.loc[prev_idx]) and (macd_line.loc[last_idx] < macd_signal.loc[last_idx])
    
    if is_bearish and (42 <= rsi.loc[last_idx] <= 60) and macd_cross_down:
        return 'SELL', current_price, atr.loc[last_idx]

    return 'HOLD', current_price, atr.loc[last_idx]

def bot_loop():
    print("🚀 बैकग्राउंड ट्रेडिंग लूप शुरू हो गया है...")
    in_position = False
    position_type = None
    stop_loss = 0
    take_profit = 0
    
    while True:
        try:
            signal, price, atr = analyze_market()
            if signal != 'HOLD' and not in_position:
                if signal == 'BUY':
                    client.create_test_order(symbol=SYMBOL, side='BUY', type='MARKET', quantity=QUANTITY)
                    stop_loss = price - (2.5 * atr)
                    take_profit = price + (4.0 * atr)
                elif signal == 'SELL':
                    client.create_test_order(symbol=SYMBOL, side='SELL', type='MARKET', quantity=QUANTITY)
                    stop_loss = price + (2.5 * atr)
                    take_profit = price - (4.0 * atr)

                in_position = True
                position_type = signal
                print(f"✅ ट्रेड एक्टिव! | SL: {stop_loss:.2f} | TP: {take_profit:.2f}")
            
            elif in_position:
                if position_type == 'BUY':
                    if price <= stop_loss:
                        client.create_test_order(symbol=SYMBOL, side='SELL', type='MARKET', quantity=QUANTITY)
                        in_position = False
                    elif price >= take_profit:
                        client.create_test_order(symbol=SYMBOL, side='SELL', type='MARKET', quantity=QUANTITY)
                        in_position = False
                elif position_type == 'SELL':
                    if price >= stop_loss:
                        client.create_test_order(symbol=SYMBOL, side='BUY', type='MARKET', quantity=QUANTITY)
                        in_position = False
                    elif price <= take_profit:
                        client.create_test_order(symbol=SYMBOL, side='BUY', type='MARKET', quantity=QUANTITY)
                        in_position = False
        except Exception as e:
            print(f"⚠️ लूप एरर: {e}")
        time.sleep(60)

if __name__ == "__main__":
    t = threading.Thread(target=bot_loop)
    t.start()
    run_web_server()

